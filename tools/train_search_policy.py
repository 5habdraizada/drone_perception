"""Train the learned search policy (sim/search_rl.py) on DroneSearchEnv's reward.

    python -m tools.train_search_policy [--iterations 60] [--workers 14] [--out models/search_policy.npz]

1. Imitation: copy the information-greedy planner from its own episodes (a sensible start).
2. PPO: improve on it with the environment's reward (bits removed + finds - false finds - energy).
Episodes run in parallel worker processes; the network update uses PyTorch.
Every few iterations the deterministic policy is scored on fixed evaluation scenes (never trained
on), and the best version is kept. Writes the weights (.npz, NumPy only at inference), a training
log (.json) and a training curve (.png).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sim.search_env import DroneSearchEnv, SearchConfig
from sim.search_policies import GreedyInfoPolicy
from sim.search_rl import N_FEATURES, LearnedPolicy, action_features, init_params

EVAL_SEEDS = range(10_000, 10_014)


def rollout(job) -> dict:
    """One episode. job = (mode, params, seed); mode "greedy" records the greedy planner's moves."""
    mode, params, seed = job
    env = DroneSearchEnv(SearchConfig(), seed=seed)
    env.reset()
    policy = GreedyInfoPolicy() if mode == "greedy" else LearnedPolicy(params, sample=(mode == "sample"), seed=seed)
    steps, total, done, trunc = [], 0.0, False, False
    while not (done or trunc):
        if mode == "greedy":
            feats, safe = action_features(env)
            a = policy(env)
            steps.append((feats, safe, a, 0.0))
        else:
            a = policy(env)
            feats, safe, a, logp = policy.last
            steps.append((feats, safe, a, logp))
        _, r, done, trunc, _ = env.step(a)
        total += r
        steps[-1] = steps[-1] + (r, done)
    info = env.info()
    info["reward"] = total
    return {"steps": steps, "info": info}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=28, help="episodes per PPO iteration")
    ap.add_argument("--imitation-episodes", type=int, default=112)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/search_policy.npz")
    args = ap.parse_args(argv)

    import torch  # training only

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in init_params(rng).items()}
    # The value estimate (training only) is a separate network on the pooled move features, so its
    # large early errors can't disturb the move scorer learned by imitation.
    vnet = {"w1": torch.randn(2 * N_FEATURES, 64) / np.sqrt(2 * N_FEATURES), "b1": torch.zeros(64),
            "w2": torch.randn(64, 1) * 0.01, "b2": torch.zeros(1)}
    for v in vnet.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(net.values(), lr=3e-4)
    vopt = torch.optim.Adam(vnet.values(), lr=1e-3)
    numpy_params = lambda: {k: v.detach().numpy().astype(np.float64) for k, v in net.items()}

    def log_probs(f, m):
        h1 = torch.tanh(f @ net["w1"] + net["b1"])
        h2 = torch.tanh(h1 @ net["w2"] + net["b2"])
        logits = (h2 @ net["w3"] + net["b3"]).squeeze(-1).masked_fill(~m, -1e9)
        return torch.log_softmax(logits, -1)

    def values(f, m):
        mf = m.unsqueeze(-1).float()
        pooled = torch.cat([(f * mf).sum(1) / mf.sum(1), (f + (mf - 1) * 1e3).max(1).values], -1)
        return (torch.tanh(pooled @ vnet["w1"] + vnet["b1"]) @ vnet["w2"] + vnet["b2"]).squeeze(-1)

    def fit_values(f, m, ret, epochs):
        for _ in range(epochs):
            perm = torch.randperm(len(ret))
            for i in range(0, len(ret), 512):
                idx = perm[i:i + 512]
                loss = ((values(f[idx], m[idx]) - ret[idx]) ** 2).mean()
                vopt.zero_grad()
                loss.backward()
                vopt.step()
        return loss.item()

    gamma, lam, clip, scale, target_kl = 0.99, 0.95, 0.1, 0.1, 0.01

    log = {"imitation": [], "ppo": [], "eval": []}
    next_seed = 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        def run(mode, n, params=None, seeds=None):
            nonlocal next_seed
            if seeds is None:
                seeds = range(next_seed, next_seed + n)
                next_seed += n
            return pool.map(rollout, [(mode, params, s) for s in seeds])

        def evaluate(it):
            eps = run("argmax", 0, numpy_params(), EVAL_SEEDS)
            row = {"iteration": it, "reward": float(np.mean([e["info"]["reward"] for e in eps])),
                   "found": int(sum(e["info"]["found"] for e in eps)),
                   "targets": int(sum(e["info"]["targets"] for e in eps)),
                   "false": int(sum(e["info"]["false_alarms"] for e in eps)),
                   "crashes": int(sum(e["info"]["crashed"] for e in eps))}
            log["eval"].append(row)
            print(f"  eval: reward {row['reward']:.1f}, found {row['found']}/{row['targets']}, "
                  f"false {row['false']}, crashes {row['crashes']}", flush=True)
            return row

        # Reference: the greedy planner on the evaluation scenes.
        t0 = time.perf_counter()
        ref = run("greedy", 0, None, EVAL_SEEDS)
        log["greedy_eval"] = {"reward": float(np.mean([e["info"]["reward"] for e in ref])),
                              "found": int(sum(e["info"]["found"] for e in ref)),
                              "targets": int(sum(e["info"]["targets"] for e in ref))}
        print(f"greedy on eval scenes: {log['greedy_eval']} ({time.perf_counter() - t0:.0f} s)", flush=True)

        # 1. Imitation of the greedy planner.
        demos = run("greedy", args.imitation_episodes)
        f = torch.tensor(np.array([s[0] for e in demos for s in e["steps"]]), dtype=torch.float32)
        m = torch.tensor(np.array([s[1] for e in demos for s in e["steps"]]))
        a = torch.tensor([s[2] for e in demos for s in e["steps"]])
        for epoch in range(30):
            perm = torch.randperm(len(a))
            losses = []
            for i in range(0, len(a), 512):
                idx = perm[i:i + 512]
                logp = log_probs(f[idx], m[idx])
                loss = -logp.gather(1, a[idx, None]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            log["imitation"].append(float(np.mean(losses)))
        with torch.no_grad():
            agree = (log_probs(f, m).argmax(1) == a).float().mean().item()
        print(f"imitation: {len(a)} decisions, loss {log['imitation'][-1]:.3f}, agrees with greedy {agree:.0%}", flush=True)
        # Fit the value estimate on the demonstrations before PPO relies on it.
        ret_demo = []
        for e in demos:
            g, rs = 0.0, []
            for s in reversed(e["steps"]):
                g = s[4] * scale + gamma * g
                rs.append(g)
            ret_demo += rs[::-1]
        mse = fit_values(f, m, torch.tensor(ret_demo, dtype=torch.float32), 40)
        print(f"value estimate fitted on the demonstrations: mse {mse:.2f}", flush=True)
        best = evaluate(0)
        best_params = numpy_params()

        # 2. PPO on the environment's reward: small steps (low learning rate, clip, KL stop).
        for group in opt.param_groups:
            group["lr"] = 1e-4
        for it in range(1, args.iterations + 1):
            t0 = time.perf_counter()
            eps = run("sample", args.episodes, numpy_params())
            f = torch.tensor(np.array([s[0] for e in eps for s in e["steps"]]), dtype=torch.float32)
            m = torch.tensor(np.array([s[1] for e in eps for s in e["steps"]]))
            a = torch.tensor([s[2] for e in eps for s in e["steps"]])
            old_logp = torch.tensor([s[3] for e in eps for s in e["steps"]], dtype=torch.float32)
            with torch.no_grad():
                vals = values(f, m).numpy()
            adv, ret, i0 = np.zeros(len(a)), np.zeros(len(a)), 0
            for e in eps:  # generalised advantage estimation, per episode
                n = len(e["steps"])
                r = np.array([s[4] for s in e["steps"]]) * scale
                v = vals[i0:i0 + n]
                last_done = e["steps"][-1][5]
                gae, nxt = 0.0, 0.0 if last_done else v[-1]
                for t in reversed(range(n)):
                    delta = r[t] + gamma * nxt - v[t]
                    gae = delta + gamma * lam * gae
                    adv[i0 + t], nxt = gae, v[t]
                ret[i0:i0 + n] = adv[i0:i0 + n] + v
                i0 += n
            adv_t = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
            ret_t = torch.tensor(ret, dtype=torch.float32)
            fit_values(f, m, ret_t, 4)
            for epoch in range(4):
                with torch.no_grad():  # stop once the policy has moved far enough from the one that acted
                    kl = (old_logp - log_probs(f, m).gather(1, a[:, None]).squeeze(1)).mean().item()
                if kl > target_kl:
                    break
                perm = torch.randperm(len(a))
                for i in range(0, len(a), 512):
                    idx = perm[i:i + 512]
                    logp_all = log_probs(f[idx], m[idx])
                    logp = logp_all.gather(1, a[idx, None]).squeeze(1)
                    ratio = torch.exp(logp - old_logp[idx])
                    pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - clip, 1 + clip) * adv_t[idx]).mean()
                    probs = logp_all.exp()
                    entropy = -(probs * logp_all.clamp(min=-30)).sum(1).mean()
                    loss = pg - 0.005 * entropy
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.values(), 0.5)
                    opt.step()
            row = {"iteration": it, "reward": float(np.mean([e["info"]["reward"] for e in eps])),
                   "found": int(sum(e["info"]["found"] for e in eps)),
                   "targets": int(sum(e["info"]["targets"] for e in eps)), "seconds": time.perf_counter() - t0}
            log["ppo"].append(row)
            print(f"iteration {it}: training reward {row['reward']:.1f}, found {row['found']}/{row['targets']} "
                  f"({row['seconds']:.0f} s)", flush=True)
            if it % args.eval_every == 0:
                score = evaluate(it)
                if score["reward"] > best["reward"]:
                    best, best_params = score, numpy_params()
                    np.savez(out, **best_params)
                    print(f"  saved best to {out}", flush=True)

    np.savez(out, **best_params)
    log["best"] = best
    out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    plot(log, out.with_suffix(".png"))
    print(f"best: {best}; log {out.with_suffix('.json')}, curve {out.with_suffix('.png')}")


def plot(log: dict, path: Path, w: int = 760, h: int = 380) -> None:
    """Training curve: sampled training reward, evaluation reward, and greedy's level (OpenCV only)."""
    import cv2

    img = np.full((h, w, 3), 255, np.uint8)
    ppo, ev, ref = log["ppo"], log["eval"], log["greedy_eval"]["reward"]
    ys = [r["reward"] for r in ppo + ev] + [ref]
    lo, hi = min(ys) - 10, max(ys) + 10
    n = max([r["iteration"] for r in ppo + ev] + [1])
    x0, x1, y0, y1 = 70, w - 20, 30, h - 50
    px = lambda it, v: (int(x0 + (x1 - x0) * it / n), int(y1 - (y1 - y0) * (v - lo) / (hi - lo)))
    cv2.line(img, (x0, y1), (x1, y1), (60, 60, 60), 1)
    cv2.line(img, (x0, y0), (x0, y1), (60, 60, 60), 1)
    for v in np.linspace(lo, hi, 5):
        cv2.putText(img, f"{v:.0f}", (8, px(0, v)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    for it in range(0, n + 1, max(1, n // 6)):
        cv2.putText(img, str(it), (px(it, lo)[0] - 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "PPO iteration (0 = after imitating greedy)", (x0 + 150, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    a, b = px(0, ref), px(n, ref)
    for xs in range(a[0], b[0], 12):
        cv2.line(img, (xs, a[1]), (min(xs + 6, b[0]), a[1]), (40, 40, 40), 1)
    pts = np.array([px(r["iteration"], r["reward"]) for r in ppo], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (190, 170, 120), 1, cv2.LINE_AA)
    pts = np.array([px(r["iteration"], r["reward"]) for r in ev], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (50, 50, 200), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(int(c) for c in p), 4, (50, 50, 200), -1, cv2.LINE_AA)
    for k, (label, col) in enumerate((("training episodes (sampled moves)", (190, 170, 120)),
                                      ("evaluation scenes (best move)", (50, 50, 200)),
                                      ("greedy planner, evaluation scenes", (40, 40, 40)))):
        cv2.line(img, (x0 + 15, 14 + 16 * k), (x0 + 40, 14 + 16 * k), col, 2)
        cv2.putText(img, label, (x0 + 48, 18 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "mean episode reward", (x1 - 170, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    main()
