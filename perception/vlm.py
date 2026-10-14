"""A small vision-language model (SmolVLM, ONNX Runtime on the CPU) as the mission advisor.

    advisor = SmolVLMAdvisor("models/smolvlm-256m")
    advisor.assess(crop, "Find vehicles. Ignore trees and roofs.")   # -> Assessment

The model folder holds the Hugging Face repo's config/tokenizer files and onnx/
{vision_encoder, embed_tokens, decoder_model_merged}[_variant].onnx (variant "" = fp32,
"int8", "fp16", ...). Needs the transformers package for the text and image processor only;
inference itself is onnxruntime, so the drone computer needs no PyTorch.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from perception.advisor import PROMPT, Assessment, parse_reply


class SmolVLMAdvisor:
    def __init__(self, model_dir: str | Path, variant: str = "", threads: int = 4,
                 max_new_tokens: int = 96, image_px: int = 512):
        import onnxruntime as ort
        from transformers import AutoConfig, AutoProcessor

        model_dir = Path(model_dir)
        self.name = f"smolvlm:{model_dir.name}{'-' + variant if variant else ''}"
        self.config = AutoConfig.from_pretrained(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        # One image tile: aerial crops are small, and every extra tile costs vision-encoder time.
        self.processor.image_processor.size = {"longest_edge": image_px}
        self.processor.image_processor.do_image_splitting = False
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        suffix = f"_{variant}" if variant else ""
        load = lambda n: ort.InferenceSession(str(model_dir / "onnx" / f"{n}{suffix}.onnx"), opts,
                                              providers=["CPUExecutionProvider"])
        self.vision, self.embed, self.decoder = load("vision_encoder"), load("embed_tokens"), load("decoder_model_merged")
        self.max_new_tokens = max_new_tokens
        self.last_seconds = 0.0
        self._stop = {self.config.text_config.eos_token_id, self.processor.tokenizer.eos_token_id}

    def assess(self, image: np.ndarray, brief: str) -> Assessment:
        return parse_reply(self.ask(image, PROMPT.format(brief=brief)), self.name)

    def ask(self, image: np.ndarray, question: str) -> str:
        """Free-form answer to a question about an image (BGR or greyscale)."""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[rgb], return_tensors="np")

        text_cfg = self.config.text_config
        past = {f"past_key_values.{layer}.{kv}": np.zeros((1, text_cfg.num_key_value_heads, 0, text_cfg.head_dim),
                                                           np.float32)
                for layer in range(text_cfg.num_hidden_layers) for kv in ("key", "value")}
        input_ids, attention = inputs["input_ids"], inputs["attention_mask"]
        positions = np.cumsum(attention, axis=-1)
        image_features, out = None, []
        for _ in range(self.max_new_tokens):
            embeds = self.embed.run(None, {"input_ids": input_ids})[0]
            if image_features is None:
                image_features = self.vision.run(["image_features"], {
                    "pixel_values": inputs["pixel_values"],
                    "pixel_attention_mask": inputs["pixel_attention_mask"].astype(np.bool_)})[0]
                embeds[inputs["input_ids"] == self.config.image_token_id] = \
                    image_features.reshape(-1, image_features.shape[-1])
            logits, *present = self.decoder.run(None, dict(inputs_embeds=embeds, attention_mask=attention,
                                                           position_ids=positions, **past))
            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            # The mask must cover the cached tokens too; a 1-element mask (as in the model card's
            # example) derails generation after the first token.
            attention = np.concatenate([attention, np.ones_like(input_ids)], axis=-1)
            positions = positions[:, -1:] + 1
            for j, key in enumerate(past):
                past[key] = present[j]
            token = int(input_ids[0, 0])
            if token in self._stop:
                break
            out.append(token)
        self.last_seconds = time.perf_counter() - t0
        return self.processor.decode(out, skip_special_tokens=True).strip()
