from collections.abc import Sequence
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MAX_EAGER_TOKENS = 1024
type Trace = dict[str, Tensor]


def chunked_attention(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    trace_start: int,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    outputs, retained = [], []
    length = query.shape[-2]
    for start in range(0, length, 256):
        stop = min(start + 256, length)
        output, weights = eager_attention_forward(
            module,
            query[..., start:stop, :],
            key[..., :stop, :],
            value[..., :stop, :],
            attention_mask[..., start:stop, :stop],
            **kwargs,
        )
        outputs.append(output)
        if stop > trace_start:
            suffix = weights[..., max(0, trace_start - start):, :]
            retained.append(F.pad(suffix, (0, length - stop)))
    return torch.cat(outputs, dim=1), torch.cat(retained, dim=-2)


AttentionInterface.register("ugar_chunked_eager", chunked_attention)
AttentionMaskInterface.register("ugar_chunked_eager", eager_mask)


def lookback_ratios(attention: Tensor, prompt_length: int) -> Tensor:
    prompt_mean = attention[..., :prompt_length].mean(dim=-1)
    lengths = torch.arange(attention.shape[1], device=attention.device)
    reply_mean = attention[..., prompt_length:].sum(dim=-1)
    reply_mean = reply_mean / lengths.clamp_min(1)
    return (prompt_mean / (prompt_mean + reply_mean).clamp_min(1e-8)).T


def recent_attention(
    attention: Tensor, prompt_length: int, history: int = 8
) -> Tensor:
    positions = torch.arange(attention.shape[1], device=attention.device)
    positions = positions[:, None] + prompt_length - 1
    indices = positions - torch.arange(history, device=attention.device)
    weights = attention.gather(
        -1, indices.clamp_min(0).expand(attention.shape[0], -1, -1)
    )
    return (weights * (indices >= 0)).transpose(0, 1).flatten(1)


def icr_scores(
    attention: Tensor, before: Tensor, after: Tensor, start: int
) -> Tensor:
    positions = torch.arange(start, after.shape[0], device=after.device)
    keys = torch.arange(after.shape[0], device=after.device)
    causal = keys <= positions[:, None]
    weights = attention.float().mean(dim=0).masked_fill(~causal, -torch.inf)
    values, indices = weights.topk(min(20, after.shape[0]), dim=-1)
    valid = torch.isfinite(values)
    delta = after[start:].float() - before[start:].to(after.device).float()
    normalized = F.normalize(after.float(), dim=-1)
    projections = torch.bmm(normalized[indices], delta.unsqueeze(-1))
    projections = projections.squeeze(-1)
    supports = valid.sum(dim=-1, keepdim=True)

    def distribution(scores: Tensor) -> Tensor:
        scores = scores.masked_fill(~valid, 0)
        mean = scores.sum(dim=-1, keepdim=True) / supports
        centered = (scores - mean) * valid
        variance = centered.square().sum(dim=-1, keepdim=True) / (
            supports - 1
        ).clamp_min(1)
        standardized = centered / variance.sqrt().clamp_min(1e-8)
        return standardized.masked_fill(~valid, -torch.inf).softmax(dim=-1)

    projection = distribution(projections)
    attended = distribution(values)
    midpoint = (projection + attended) / 2
    log_mid = midpoint.clamp_min(1e-12).log()
    first = projection * (projection.clamp_min(1e-12).log() - log_mid)
    second = attended * (attended.clamp_min(1e-12).log() - log_mid)
    return ((first + second).sum(dim=-1) / 2).clamp_min(0)


class TransformersExtractor:
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, model: PreTrainedModel
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.layers = self.model.base_model.layers
        self.saplma_layer = max(0, len(self.layers) - 5)
        self.haloscope_layer = (len(self.layers) - 1) // 2

    @classmethod
    def from_pretrained(
        cls, model_id: str = MODEL_ID, quantize: bool = True
    ) -> "TransformersExtractor":
        revision = MODEL_REVISION if model_id == MODEL_ID else None
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        cuda = torch.cuda.is_available()
        if quantize and not cuda:
            raise RuntimeError("Use quantize=False for CPU inference.")
        options = {}
        if quantize:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map="auto",
            dtype=torch.float16 if cuda else torch.float32,
            attn_implementation="eager",
            **options,
        )
        return cls(tokenizer, model)

    def feature_dim(self) -> int:
        return self.model.config.hidden_size

    def output_attention(self) -> bool:
        return True

    def _prompt(self, prompt: str) -> BatchEncoding:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            tokenizer_kwargs={"return_token_type_ids": False},
        ).to(self.model.device)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self._prompt(prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def __call__(self, prompt: str, reply: str) -> Trace:
        ids, boundary = self._encode(prompt, reply)
        return self.extract_ids(ids, boundary)

    def _encode(self, prompt: str, reply: str) -> tuple[list[int], int]:
        prefix = self._prompt(prompt)["input_ids"][0].tolist()
        suffix = self.tokenizer.encode(reply, add_special_tokens=False)
        if not prefix or not suffix:
            raise ValueError("The prompt and reply must both contain tokens.")
        return prefix + suffix, len(prefix)

    @torch.no_grad()
    def hidden_states(
        self, prompt: str, reply: str, layers: Sequence[int] = (-1,)
    ) -> Tensor:
        ids, boundary = self._encode(prompt, reply)
        outputs = self.model.base_model(
            input_ids=torch.tensor([ids], device=self.model.device),
            use_cache=False,
            output_hidden_states=True,
        )
        return torch.cat(
            [outputs.hidden_states[layer][0, boundary - 1:-1].cpu()
             for layer in layers],
            dim=-1,
        )

    @torch.no_grad()
    def extract_ids(
        self, input_ids: Sequence[int], prompt_length: int
    ) -> Trace:
        if not 0 < prompt_length < len(input_ids):
            raise ValueError("The prompt and reply must both contain tokens.")
        prompt_end = prompt_length - 1
        query_count = len(input_ids) - prompt_end
        lookback, recent, icr = [], [], []
        hidden: Trace = {}
        current_attention: Tensor | None = None

        def capture_attention(
            module: nn.Module, args: tuple, result: tuple[Tensor, Tensor]
        ) -> None:
            nonlocal current_attention
            current_attention = result[1][0, :, -query_count:, :]

        def capture_layer(
            index: int, module: nn.Module, args: tuple, result: Tensor
        ) -> None:
            nonlocal current_attention
            before = args[0][0]
            after = result[0]
            attention = current_attention
            if attention is None:
                raise RuntimeError("The model must expose eager attention.")
            attention = attention.to(after.device)
            prediction_attention = attention[:, :-1]
            lookback.append(
                lookback_ratios(
                    prediction_attention.float(), prompt_length
                ).cpu()
            )
            recent.append(
                recent_attention(
                    prediction_attention, prompt_length
                ).float().cpu()
            )
            scores = icr_scores(attention[:, 1:], before, after, prompt_length)
            icr.append(scores.cpu())
            if index == self.saplma_layer:
                hidden["saplma"] = after[prompt_length:].to(
                    "cpu", torch.float16
                )
            if index == self.haloscope_layer:
                hidden["haloscope"] = after[prompt_length:].to(
                    "cpu", torch.float16
                )
            current_attention = None

        hooks = []
        options = {}
        original_attention = self.model.config._attn_implementation
        chunked = len(input_ids) > MAX_EAGER_TOKENS
        try:
            if chunked:
                self.model.set_attn_implementation("ugar_chunked_eager")
                options["trace_start"] = prompt_end
            for index, layer in enumerate(self.layers):
                hooks.append(
                    layer.self_attn.register_forward_hook(capture_attention)
                )
                hooks.append(
                    layer.register_forward_hook(partial(capture_layer, index))
                )
            ids = torch.tensor([input_ids], device=self.model.device)
            self.model.base_model(input_ids=ids, use_cache=False, **options)
        finally:
            for hook in hooks:
                hook.remove()
            if chunked:
                self.model.set_attn_implementation(original_attention)

        return {
            **hidden,
            "lookback": torch.cat(lookback, dim=-1),
            "recent_attention": torch.stack(recent).amax(dim=0),
            "icr": torch.stack(icr, dim=-1),
        }
