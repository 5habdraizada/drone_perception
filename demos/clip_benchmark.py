"""Compare zero-shot image-text models as mission advisors on real aerial crops (Delft, 8 cm).

    python -m demos.clip_benchmark [--area data/delft] [--crop-m 10]

Labelled set: vehicle positions picked by eye from the orthophoto, plus non-vehicle crops
labelled from the BGT (the Dutch surveyed topographic map: buildings, water, roads, green
space) and LiDAR heights (roof = inside a building and raised; trees = tall, outside
buildings). Mixed crops are skipped. Needs tools/fetch_pdok.py --bgt. Contact sheets are
saved so every label can be checked.

Two tasks:
  vehicle vs not   the steering decision ("Find vehicles"): accuracy, precision, recall, AUC
  6-way land cover vehicle / roof / trees / grass / road / water: top-1 accuracy

Models (both ViT-B/32, both timed through ONNX Runtime on the CPU):
  clip         OpenAI CLIP, ONNX export from Hugging Face (Xenova/clip-vit-base-patch32)
  remoteclip   RemoteCLIP (trained on remote-sensing images), exported to ONNX here
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Vehicle centres in full-resolution orthophoto pixels (north-up), found by eye.
VEHICLES = [
    (240, 50), (85, 115), (40, 290), (45, 415), (110, 430), (75, 485), (140, 515), (165, 550), (20, 570),
    (155, 630), (195, 655), (165, 720), (85, 770), (185, 480),                        # car park, top left
    (3662, 71), (3511, 142), (3413, 177), (3316, 230), (3218, 257), (3121, 301), (3041, 328), (2970, 363),
    (2882, 390),                                                                       # parked row, top road
    (525, 2805), (161, 2955), (525, 3541), (559, 3644), (157, 3691), (464, 3678),     # bottom left
    (2661, 3095), (2723, 2318),                                                        # tram, van
]
CLASSES = {
    "vehicle": ["cars", "a car", "parked cars", "a van", "a bus", "a tram"],
    "roof": ["a building roof", "a flat roof", "rooftops"],
    "trees": ["trees", "tree canopy"],
    "grass": ["grass", "a lawn"],
    "road": ["a road", "pavement", "a footpath"],
    "water": ["water", "a canal"],
}
TEMPLATES = ["an aerial photo of {}.", "a satellite image of {}.", "a top-down view of {}."]


def build_dataset(area: Path, crop_m: float, per_class: int, seed: int = 0):
    meta = json.loads((area / "area.json").read_text())
    ortho = cv2.imread(str(area / "ortho.png"))
    res = (meta["bbox"][2] - meta["bbox"][0]) / ortho.shape[0]
    half = int(round(crop_m / res / 2))
    dsm = cv2.imread(str(area / "dsm.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64)
    dtm = cv2.imread(str(area / "dtm.tif"), cv2.IMREAD_UNCHANGED).astype(np.float64)
    water = dsm > 1e30
    dsm[water] = np.nan
    dtm[dtm > 1e30] = np.nan
    ground = np.nanmedian(dtm)
    ndsm = np.where(np.isnan(dtm), dsm - ground, dsm - dtm)  # under buildings the ground model is empty
    k = ortho.shape[0] / dsm.shape[0]                          # ortho pixels per DEM cell

    def crop(x, y):
        return ortho[y - half: y + half, x - half: x + half]

    items = [(crop(x, y), "vehicle", x, y) for x, y in VEHICLES if half <= x < ortho.shape[1] - half
             and half <= y < ortho.shape[0] - half]

    # Land cover from the BGT (surveyed topography) plus LiDAR heights, on the orthophoto grid.
    bbox = meta["bbox"]
    bgt = lambda coll, keep=lambda p: True: _rasterise(area / f"bgt_{coll}.json", bbox, ortho.shape[:2], res, keep)
    building, water_m = bgt("pand"), bgt("waterdeel")
    road = bgt("wegdeel", keep=lambda p: "parkeervlak" not in (p.get("functie") or ""))  # parking bays hold cars
    green = bgt("begroeidterreindeel")
    h = cv2.resize(np.nan_to_num(ndsm, nan=0.0), ortho.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    tall, flat = h > 3.0, np.abs(h) < 0.5

    rng = np.random.default_rng(seed)
    counts = {c: 0 for c in CLASSES if c != "vehicle"}
    tries = 0
    while min(counts.values()) < per_class and tries < 50000:
        tries += 1
        x, y = (int(v) for v in rng.integers(half, ortho.shape[0] - half, size=2))
        if min(np.hypot(x - vx, y - vy) for vx, vy in VEHICLES) < 2.2 * half:
            continue  # keep clear of known vehicles
        win = (slice(y - half, y + half), slice(x - half, x + half))
        f = {name: m[win].mean() for name, m in
             (("building", building), ("water", water_m), ("road", road), ("green", green), ("tall", tall),
              ("flat", flat))}
        if f["building"] > 0.95 and h[win].mean() > 2.5:
            label = "roof"
        elif f["water"] > 0.9:
            label = "water"
        elif f["tall"] > 0.7 and f["building"] < 0.05 and f["green"] > 0.6:
            # Green space too: the photo isn't a true orthophoto, so tall roofs appear shifted
            # off their footprints and would otherwise pass as "tall, outside buildings".
            label = "trees"
        elif f["green"] > 0.9 and f["flat"] > 0.9:
            label = "grass"
        elif f["road"] > 0.9 and f["flat"] > 0.9:
            label = "road"
        else:
            continue  # mixed crops are skipped, not guessed
        if counts[label] < per_class:
            counts[label] += 1
            items.append((crop(x, y), label, x, y))
    return items


def _rasterise(path: Path, bbox, shape, res: float, keep) -> np.ndarray:
    """BGT polygons (RD coordinates) -> boolean mask on the north-up orthophoto grid."""
    mask = np.zeros(shape, np.uint8)
    if not path.exists():
        raise SystemExit(f"{path} missing: run python -m tools.fetch_pdok --bgt ... first")
    for feat in json.loads(path.read_text())["features"]:
        if not keep(feat.get("properties") or {}):
            continue
        geom = feat.get("geometry") or {}
        polys = geom.get("coordinates", []) if geom.get("type") == "MultiPolygon" else (
            [geom["coordinates"]] if geom.get("type") == "Polygon" else [])
        for poly in polys:
            for i, ring in enumerate(poly):  # first ring is the outline, the rest are holes
                pts = np.array([[(p[0] - bbox[0]) / res, (bbox[3] - p[1]) / res] for p in ring], np.int32)
                cv2.fillPoly(mask, [pts], 1 if i == 0 else 0)
    return mask.astype(bool)


def contact_sheet(items, path: Path, size: int = 120) -> None:
    tiles = []
    for img, label, x, y in items:
        t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (size, 14), (0, 0, 0), -1)
        cv2.putText(t, f"{label} {x},{y}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


class OnnxClip:
    """Image and text towers as ONNX sessions; text prompts are embedded once."""

    def __init__(self, name, vision_path, text_fn, mean, std, size=224, threads=4):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.name, self.size = name, size
        self.vision = ort.InferenceSession(str(vision_path), opts, providers=["CPUExecutionProvider"])
        self.input = self.vision.get_inputs()[0].name
        self.mean, self.std = np.array(mean, np.float32), np.array(std, np.float32)
        self.text_fn = text_fn

    def image_embedding(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255 - self.mean) / self.std).transpose(2, 0, 1)[None]
        e = self.vision.run(None, {self.input: x})[0].reshape(-1)
        return e / np.linalg.norm(e)

    def class_embeddings(self) -> np.ndarray:
        rows = []
        for names in CLASSES.values():
            e = self.text_fn([t.format(n) for n in names for t in TEMPLATES])
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            m = e.mean(axis=0)
            rows.append(m / np.linalg.norm(m))
        return np.array(rows)


def load_clip(model_dir: Path) -> OnnxClip:
    import onnxruntime as ort
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(model_dir)
    text = ort.InferenceSession(str(model_dir / "onnx" / "text_model.onnx"), providers=["CPUExecutionProvider"])
    names = [i.name for i in text.get_inputs()]

    def text_fn(prompts):
        t = tok(prompts, padding=True, return_tensors="np")
        feeds = {n: t[n].astype(np.int64) for n in names}
        outs = {o.name: v for o, v in zip(text.get_outputs(), text.run(None, feeds))}
        return outs.get("text_embeds", next(iter(outs.values())))

    vision = model_dir / "onnx" / "vision_model.onnx"
    clip = OnnxClip("clip", vision, text_fn, (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
    out_names = [o.name for o in clip.vision.get_outputs()]
    if "image_embeds" in out_names:  # use the projected embedding, not the raw pooled output
        clip.image_embedding = _named_output(clip, "image_embeds")
    return clip


def _named_output(clip: OnnxClip, output: str):
    def embed(bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (clip.size, clip.size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255 - clip.mean) / clip.std).transpose(2, 0, 1)[None]
        e = clip.vision.run([output], {clip.input: x})[0].reshape(-1)
        return e / np.linalg.norm(e)
    return embed


def load_remoteclip(weights: Path, onnx_out: Path) -> OnnxClip:
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    model.eval()
    if not onnx_out.exists():
        torch.onnx.export(model.visual, torch.randn(1, 3, 224, 224), str(onnx_out), input_names=["pixel_values"],
                          output_names=["image_embeds"], opset_version=17, dynamo=False)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def text_fn(prompts):
        with torch.no_grad():
            return model.encode_text(tokenizer(prompts)).numpy()

    return OnnxClip("remoteclip", onnx_out, text_fn, (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))


def evaluate(model: OnnxClip, items) -> dict:
    classes = list(CLASSES)
    text = model.class_embeddings()
    times, probs = [], []
    model.image_embedding(items[0][0])  # warm-up
    for img, *_ in items:
        t0 = time.perf_counter()
        e = model.image_embedding(img)
        times.append(time.perf_counter() - t0)
        logits = 100.0 * text @ e
        p = np.exp(logits - logits.max())
        probs.append(p / p.sum())
    probs = np.array(probs)
    truth = np.array([classes.index(it[1]) for it in items])
    pred = probs.argmax(axis=1)
    is_veh = truth == 0
    score = probs[:, 0]
    said = score >= 0.5
    tp, fp, fn = int((said & is_veh).sum()), int((said & ~is_veh).sum()), int((~said & is_veh).sum())
    return {"top1": float((pred == truth).mean()),
            "per_class": {c: float((pred[truth == i] == i).mean()) for i, c in enumerate(classes) if (truth == i).any()},
            "vehicle_acc": float((said == is_veh).mean()),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "auc": _auc(score[is_veh], score[~is_veh]),
            "ms": 1000 * float(np.median(times)), "pred": pred, "probs": probs}


def _auc(pos, neg) -> float:
    """Probability a random vehicle crop scores above a random non-vehicle crop."""
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="data/delft")
    ap.add_argument("--crop-m", type=float, default=10.0)
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--out", default="out/clip_benchmark")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = build_dataset(Path(args.area), args.crop_m, args.per_class)
    counts = {c: sum(1 for it in items if it[1] == c) for c in CLASSES}
    print(f"{len(items)} labelled {args.crop_m:g} m crops: {counts}")
    for c in CLASSES:
        contact_sheet([it for it in items if it[1] == c], out / f"sheet_{c}.jpg")

    models = [load_clip(Path("models/clip-vit-b32")),
              load_remoteclip(Path("models/remoteclip/RemoteCLIP-ViT-B-32.pt"), Path("models/remoteclip/vision.onnx"))]
    results = {m.name: evaluate(m, items) for m in models}

    print(f"\n{'':12s} {'vehicle acc':>11s} {'precision':>9s} {'recall':>6s} {'AUC':>5s} {'6-way top-1':>11s} {'ms/crop':>8s}")
    for name, r in results.items():
        print(f"{name:12s} {r['vehicle_acc']:11.0%} {r['precision']:9.0%} {r['recall']:6.0%} {r['auc']:5.2f} "
              f"{r['top1']:11.0%} {r['ms']:8.1f}")
    print("\nper-class top-1:")
    for c in CLASSES:
        print(f"  {c:8s} n={counts[c]:2d}  " + "  ".join(f"{n} {r['per_class'].get(c, float('nan')):4.0%}"
                                                    for n, r in results.items()))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label"] + [f"{n}_pred" for n in results] + [f"{n}_p_vehicle" for n in results])
        classes = list(CLASSES)
        for i, (_, label, x, y) in enumerate(items):
            w.writerow([x, y, label] + [classes[r["pred"][i]] for r in results.values()]
                       + [round(float(r["probs"][i, 0]), 3) for r in results.values()])
    print(f"\ncontact sheets and predictions in {out}/")


if __name__ == "__main__":
    main()
