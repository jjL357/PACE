"""lmms-eval adapter for PACE on Qwen2.5-VL."""

from __future__ import annotations

import math
import time
from typing import List

from loguru import logger
from tqdm import tqdm

import lmms_eval.models.simple.qwen2_5_vl as upstream_qwen_adapter
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
from lmms_eval.protocol import ChatMessages
from qwen_vl_utils import process_vision_info

from .adaptive_pixel_compressor import AdaptivePixelCompressor, resize_to_pixel_bounds
from .modeling_pace_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


class PACEQwen2_5VL(upstream_qwen_adapter.Qwen2_5_VL):
    """PACE inference backend exposed as ``pace_qwen2_5_vl`` in lmms-eval."""

    is_simple = False

    def __init__(self, **kwargs) -> None:
        token_budget = float(kwargs.pop("token_budget", 0.10))
        extraction_layer = int(kwargs.pop("extraction_layer", 2))
        fusion_temperature = float(kwargs.pop("fusion_temperature", 0.5))
        input_min_pixels = int(kwargs.pop("input_min_pixels", 2048 * 28 * 28))
        input_max_pixels = int(kwargs.pop("input_max_pixels", 2048 * 28 * 28))
        apc_preview_depth = int(kwargs.pop("apc_preview_depth", 1))
        apc_global_weight = float(kwargs.pop("apc_global_weight", 0.6))
        apc_detail_fraction = float(kwargs.pop("apc_detail_fraction", 0.1))
        apc_detail_scale = float(kwargs.pop("apc_detail_scale", 1.5))
        apc_minimum_retention = float(kwargs.pop("apc_minimum_retention", 0.05))
        use_apc = self._parse_bool(kwargs.pop("use_apc", True))

        if not 0.0 < token_budget <= 1.0:
            raise ValueError("token_budget must be in (0, 1].")
        if fusion_temperature <= 0.0:
            raise ValueError("fusion_temperature must be positive.")
        if input_min_pixels > input_max_pixels:
            raise ValueError("input_min_pixels cannot exceed input_max_pixels.")

        # The upstream adapter has no model-class injection point. Replace its
        # module-level class only while it constructs this instance.
        original_model_class = upstream_qwen_adapter.Qwen2_5_VLForConditionalGeneration
        upstream_qwen_adapter.Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration
        )
        try:
            super().__init__(**kwargs)
        finally:
            upstream_qwen_adapter.Qwen2_5_VLForConditionalGeneration = original_model_class

        self.token_budget = token_budget
        self.input_min_pixels = input_min_pixels
        self.input_max_pixels = input_max_pixels
        self.use_apc = use_apc
        self.model.config.target_layer_id = extraction_layer
        self.model.config.ddae_fusion_temperature = fusion_temperature
        self.apc = AdaptivePixelCompressor(
            vision_model=self.model.visual,
            preview_depth=apc_preview_depth,
            global_weight=apc_global_weight,
            detail_fraction=apc_detail_fraction,
            detail_scale=apc_detail_scale,
            minimum_retention=apc_minimum_retention,
            patch_size=28,
        )
        logger.info(
            "PACE initialized: budget={}, APC={}, preview_depth={}, "
            "extraction_layer={}, fusion_temperature={}",
            self.token_budget,
            self.use_apc,
            apc_preview_depth,
            extraction_layer,
            fusion_temperature,
        )

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value {value!r}.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate deterministic responses for lmms-eval chat tasks."""
        responses = []
        adaptive_ratios = []
        end_to_end_latency = 0.0
        generated_token_count = 0

        def collate_key(item):
            return item[0], item[0]

        reordered = utils.Collator(
            [request.args for request in requests],
            collate_key,
            group_fn=lambda item: item[2],
            grouping=True,
        )
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)
        progress = tqdm(
            total=math.ceil(len(requests) / self.batch_size),
            disable=self.rank != 0,
            desc="PACE responding",
        )

        for chunk in chunks:
            _, doc_to_messages, generation_kwargs, doc_ids, tasks, splits = zip(*chunk)
            messages = [
                doc_to_messages[index](self.task_dict[task][split][doc_id])
                for index, (doc_id, task, split) in enumerate(zip(doc_ids, tasks, splits))
            ]
            chat_messages = [ChatMessages(messages=message) for message in messages]
            hf_messages = [message.to_hf_messages() for message in chat_messages]
            prompts = self.processor.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(hf_messages)
            if video_inputs:
                raise NotImplementedError(
                    "This release targets the image benchmarks reported in the paper."
                )
            if not image_inputs or len(image_inputs) != 1:
                raise ValueError("PACE currently requires exactly one image per batch.")

            bounded_image = resize_to_pixel_bounds(
                image_inputs[0],
                min_pixels=self.input_min_pixels,
                max_pixels=self.input_max_pixels,
            )
            if self.use_apc:
                apc_result = self.apc.compress(
                    bounded_image,
                    processor=self.processor,
                    device=self.device,
                )
                image_inputs = [apc_result.image]
                adaptive_ratio = apc_result.retention_ratio
            else:
                image_inputs = [bounded_image]
                adaptive_ratio = 1.0
            adaptive_ratios.append(adaptive_ratio)

            # DDAE receives the remaining fraction needed to satisfy the global
            # visual-token budget after APC has condensed the pixel sequence.
            ddae_ratio = min(self.token_budget / max(adaptive_ratio, 1e-6), 1.0)
            self.model.config.budget = [ddae_ratio]

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=prompts,
                images=image_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

            requested_generation = generation_kwargs[0]
            generation_config = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
                **requested_generation,
            }
            generation_config.pop("until", None)
            do_sample = generation_config["temperature"] > 0
            generation_config["do_sample"] = do_sample
            generation_config["use_cache"] = self.use_cache
            if not do_sample:
                generation_config["temperature"] = None
                generation_config["top_p"] = None
                generation_config["top_k"] = None

            start_time = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_config,
            )
            end_to_end_latency += time.perf_counter() - start_time

            generated_ids = [
                output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, output_ids)
            ]
            generated_token_count += sum(len(token_ids) for token_ids in generated_ids)
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for answer, prompt in zip(decoded, prompts):
                answer = parse_reasoning_model_answer(answer)
                responses.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (prompt, requested_generation), answer
                )
            progress.update(1)

        progress.close()
        responses = reordered.get_original(responses)
        average_ratio = sum(adaptive_ratios) / len(adaptive_ratios) if adaptive_ratios else 1.0
        log_metrics(
            total_tokens=generated_token_count,
            e2e_latency=end_to_end_latency,
            avg_speed=(
                generated_token_count / end_to_end_latency if end_to_end_latency > 0 else 0.0
            ),
            additional_metrics={
                "rank": self.rank,
                "average_apc_retention": average_ratio,
                "token_budget": self.token_budget,
            },
        )
        return responses
