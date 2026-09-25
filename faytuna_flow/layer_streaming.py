"""Layer-by-Layer FP16/FP32 VRAM Streaming Engine for Qwen 3.5.

Enables full unquantized FP16/FP32 forward passes of 0.8B and 4B models on consumer GPUs
(e.g., AMD Radeon RX 580 via DirectML) by loading one layer at a time into VRAM (~50 MB for 0.8B,
~250 MB for 4B), processing an entire prompt batch through that layer, offloading back to host RAM,
and streaming sequentially:
    H_0 -> Layer 0 -> H_1 -> Layer 1 -> ... -> Layer L-1 -> H_L

Features:
1. Gated DeltaNet DirectML Patch: Resolves DirectML crashes on 5D tensor .tril() and in-place
   slice mutations by using broadcasted 2D triangle masks and row-wise accumulation with torch.stack.
2. Dynamic Batching & Unpadding: Sequences of varying lengths are batched with attention masks
   and unpadded back to exact lengths.
3. Universal Architecture Support: Supports HuggingFace checkpoints (Qwen 3.5) and synthetic transformer
   models for unit testing.
4. Zero Quantization: Operates strictly in FP32/FP16 without loss of fidelity.
"""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_execution_device(requested: str | torch.device = "auto") -> tuple[torch.device, str]:
    """Resolve target execution device (including DirectML) with automatic CPU fallback.

    Args:
        requested: Device name ("auto", "dml", "directml", "cuda", "cpu", or torch.device).

    Returns:
        Tuple of (torch.device or DirectML device, device_type_str).
    """
    if isinstance(requested, torch.device):
        return requested, str(requested.type)

    req_lower = str(requested).lower().strip()
    if req_lower in {"dml", "directml"}:
        try:
            import torch_directml

            return torch_directml.device(), "dml"
        except Exception as err:
            print(f"[resolve_device] Failed to initialize DirectML ({err}); falling back to CPU.")
            return torch.device("cpu"), "cpu"
    elif req_lower == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml

            return torch_directml.device(), "dml"
        except Exception:
            return torch.device("cpu"), "cpu"
    else:
        try:
            return torch.device(requested), req_lower
        except Exception:
            return torch.device("cpu"), "cpu"


def _safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Stable L2-normalization for query and key in linear attention."""
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    return x / (norm + eps)


def patched_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """DirectML-compatible chunked Gated DeltaNet linear attention kernel.

    Fixes:
    1. Multi-dimensional .tril() failure on DirectML: Replaces 5D .tril() calls with
       a 2D broadcasted tril_mask (`diff = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_mask`).
    2. In-place slice mutation failure on DirectML: Replaces `attn[..., i, :i] = ...` slice
       assignments with row-by-row accumulation and `torch.stack` to avoid silent DirectML drops.
    3. DirectML torch.eye failure: Constructs the identity matrix on CPU and transfers to device.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _safe_l2norm(query, dim=-1, eps=1e-6)
        key = _safe_l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Chunk decay with DirectML 2D broadcasted mask
    g = g.cumsum(dim=-1)
    tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device))
    diff = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_mask
    decay_mask = diff.exp() * tril_mask
    strict_tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device), diagonal=-1)

    M = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_tril_mask

    # Row-by-row accumulation using torch.stack to bypass DirectML in-place strided slice bug
    rows: list[torch.Tensor] = [torch.zeros((*M.shape[:-2], chunk_size), device=M.device, dtype=M.dtype)]
    for i in range(1, chunk_size):
        row_init = M[..., i, :i]
        sub = torch.stack(rows[:i], dim=-2)[..., :i]
        new_row_prefix = row_init + (row_init.unsqueeze(-1) * sub).sum(-2)
        pad = torch.zeros((*M.shape[:-2], chunk_size - i), device=M.device, dtype=M.dtype)
        new_row = torch.cat([new_row_prefix, pad], dim=-1)
        rows.append(new_row)
    attn = torch.stack(rows, dim=-2)

    # Identity matrix initialized on CPU and moved to device to avoid aten::eye DirectML bug
    eye = torch.eye(chunk_size, dtype=attn.dtype).to(attn.device)
    attn = attn + eye

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]) * tril_mask
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_chunk @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_gated_deltanet_patch(target: Any | None = None) -> None:
    """Apply DirectML-compatible Gated DeltaNet patch to transformers module and model instances.

    Args:
        target: Optional model or module to patch. If None, patches global transformers module.
    """
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

        qmod.torch_chunk_gated_delta_rule = patched_chunk_gated_delta_rule
    except (ImportError, AttributeError):
        pass

    if target is not None:
        layers = getattr(target, "layers", None)
        if layers is None and hasattr(target, "model"):
            layers = getattr(target.model, "layers", None)
        if layers is not None:
            for layer in layers:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn.chunk_gated_delta_rule = patched_chunk_gated_delta_rule


class StreamOutput(list):
    """Container for streamed hidden states.

    Behaves as a list of dicts: `output[prompt_idx][layer_idx] -> np.ndarray`.
    For single-prompt calls, also supports direct dict-like access: `output[layer_idx] -> np.ndarray`.
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, np.integer)):
            if 0 <= key < len(self):
                return super().__getitem__(key)
            if len(self) == 1 and key in self[0]:
                return self[0][key]
        return super().__getitem__(key)

    def get(self, key: int, default: Any = None) -> Any:
        """Dict-like get for single-prompt extraction."""
        if len(self) == 1 and key in self[0]:
            return self[0][key]
        return default


class PipelinedLayerStreamer:
    """Layer-by-Layer VRAM Streaming Engine for unquantized Qwen 3.5 and transformer models.

    Maintains model weights in host memory (RAM) and streams one layer at a time into GPU VRAM
    (DirectML or CUDA) in FP16/FP32, processes the prompt batch, offloads the layer back to host RAM,
    and returns exact unpadded hidden states for all layer boundaries.
    """

    def __init__(
        self,
        model_or_path: str | Path | nn.Module,
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype | None = None,
        tokenizer: Any | None = None,
        chunk_size: int = 64,
    ) -> None:
        """Initialize streamer from HuggingFace checkpoint path or existing PyTorch module.

        Args:
            model_or_path: Path to model directory or pre-instantiated PyTorch model.
            device: Target execution device ("auto", "dml", "cuda", "cpu").
            dtype: Floating point precision (torch.float32 or torch.float16).
            tokenizer: Optional pre-loaded tokenizer.
            chunk_size: Chunk size for chunked linear attention (default: 64).
        """
        self.target_device, self.device_type = resolve_execution_device(device)
        self.chunk_size = chunk_size
        self.model_path: Path | None = None

        # Determine target dtype: DirectML requires FP32 or FP16 (bfloat16 causes hard crash)
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype

        apply_gated_deltanet_patch()

        if isinstance(model_or_path, (str, Path)):
            self.model_path = Path(model_or_path)
            self._init_from_path(self.model_path, tokenizer=tokenizer)
        elif isinstance(model_or_path, nn.Module):
            self._init_from_module(model_or_path, tokenizer=tokenizer)
        else:
            raise TypeError(f"Unsupported model_or_path type: {type(model_or_path)}")

        # Ensure all model parameters and buffers are converted from bfloat16 to self.dtype
        self._sanitize_dtypes()
        apply_gated_deltanet_patch(self.model)

    def _init_from_path(self, path: Path, tokenizer: Any | None = None) -> None:
        """Load model components from directory path on host CPU."""
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            except Exception:
                self.tokenizer = None

        # Load model on CPU with low_cpu_mem_usage
        load_dtype = torch.float32 if self.device_type == "dml" else self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path),
            dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        backbone = getattr(self.model, "model", getattr(self.model, "transformer", self.model))
        self.embed_tokens = getattr(backbone, "embed_tokens", getattr(backbone, "wte", None))
        self.rotary_emb = getattr(backbone, "rotary_emb", None)
        self.layers = getattr(backbone, "layers", getattr(backbone, "h", None))
        self.norm = getattr(backbone, "norm", getattr(backbone, "ln_f", None))
        self.config = getattr(self.model, "config", None)

    def _init_from_module(self, module: nn.Module, tokenizer: Any | None = None) -> None:
        """Wrap existing PyTorch model module."""
        self.tokenizer = tokenizer
        self.model = module
        self.model.eval()

        backbone = getattr(module, "model", getattr(module, "transformer", module))
        self.embed_tokens = getattr(backbone, "embed_tokens", getattr(backbone, "wte", None))
        self.rotary_emb = getattr(backbone, "rotary_emb", None)
        self.layers = getattr(backbone, "layers", getattr(backbone, "h", None))
        self.norm = getattr(backbone, "norm", getattr(backbone, "ln_f", None))
        self.config = getattr(module, "config", None)

    def _sanitize_dtypes(self) -> None:
        """Convert any unsupported bfloat16 parameters and buffers to self.dtype."""
        for p in self.model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(self.dtype)
        for b in self.model.buffers():
            if b.dtype == torch.bfloat16:
                b.data = b.data.to(self.dtype)

    @property
    def num_layers(self) -> int:
        """Total number of decoder layers in the model."""
        if self.layers is not None:
            return len(self.layers)
        return 0

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | Sequence[str] | Sequence[Sequence[int]],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Normalize input_ids, attention_mask, and compute unpadded sequence lengths."""
        if isinstance(input_ids, (list, tuple)) and len(input_ids) > 0 and isinstance(input_ids[0], str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required when passing string prompts to stream_forward.")
            tok_out = self.tokenizer(
                list(input_ids),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            t_input_ids = tok_out["input_ids"]
            t_attention_mask = tok_out["attention_mask"]
        elif isinstance(input_ids, (list, tuple)):
            max_len = max(len(s) for s in input_ids)
            padded = [list(s) + [0] * (max_len - len(s)) for s in input_ids]
            masks = [[1] * len(s) + [0] * (max_len - len(s)) for s in input_ids]
            t_input_ids = torch.tensor(padded, dtype=torch.long)
            t_attention_mask = torch.tensor(masks, dtype=torch.long) if attention_mask is None else attention_mask
        elif isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 1:
                t_input_ids = input_ids.unsqueeze(0)
            else:
                t_input_ids = input_ids
            if attention_mask is None:
                t_attention_mask = torch.ones_like(t_input_ids)
            else:
                t_attention_mask = attention_mask if attention_mask.ndim == 2 else attention_mask.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported input_ids type: {type(input_ids)}")

        unpadded_lens = t_attention_mask.sum(dim=1).tolist()
        return t_input_ids, t_attention_mask, [int(x) for x in unpadded_lens]

    def _prepare_masks(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        text_position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute causal mask and linear attention mask for Qwen 3.5 architecture."""
        causal_mask = None
        linear_attn_mask = None

        try:
            import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

            if hasattr(qmod, "create_causal_mask") and self.config is not None:
                causal_mask = qmod.create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    position_ids=text_position_ids,
                )
        except Exception:
            causal_mask = None

        if attention_mask is not None and not torch.all(attention_mask == 1):
            linear_attn_mask = attention_mask
        else:
            linear_attn_mask = None

        return causal_mask, linear_attn_mask

    def stream_forward(
        self,
        input_ids: torch.Tensor | Sequence[str] | Sequence[Sequence[int]],
        attention_mask: torch.Tensor | None = None,
        target_layers: Sequence[int] | None = None,
        device: str | torch.device | None = None,
    ) -> StreamOutput:
        """Execute layer-by-layer VRAM forward pass across a batch of prompts.

        Args:
            input_ids: Input tokens as Tensor (batch_size, seq_len) or Sequence of string prompts.
            attention_mask: Optional attention mask (batch_size, seq_len).
            target_layers: Optional sequence of layer indices to extract. If None, extracts all layers.
            device: Execution device for active layer. Defaults to self.target_device.

        Returns:
            StreamOutput containing a dict of {layer_idx: np.ndarray} for each prompt.
        """
        if device is not None:
            exec_dev, exec_dev_type = resolve_execution_device(device)
        else:
            exec_dev, exec_dev_type = self.target_device, self.device_type

        t_ids, t_mask, unpadded_lens = self._prepare_inputs(input_ids, attention_mask)
        batch_size, seq_len = t_ids.shape

        padding_side = "right"
        if self.tokenizer is not None and getattr(self.tokenizer, "padding_side", None) is not None:
            padding_side = self.tokenizer.padding_side

        target_set = set(target_layers) if target_layers is not None else None
        num_layers = self.num_layers

        # Initialize per-prompt output storage
        prompt_outputs: list[dict[int, np.ndarray]] = [{} for _ in range(batch_size)]

        with torch.no_grad():
            # Step 1: Initial Embedding on host CPU
            if self.embed_tokens is not None:
                h = self.embed_tokens(t_ids.to(self.embed_tokens.weight.device))
            else:
                # Fallback for models without explicit embed_tokens
                h = torch.zeros((batch_size, seq_len, 32), dtype=self.dtype)

            # Store Layer 0 (Embedding Output)
            if target_set is None or 0 in target_set:
                for b_idx, plen in enumerate(unpadded_lens):
                    if padding_side == "left":
                        h_valid = h[b_idx, -plen:]
                    else:
                        h_valid = h[b_idx, :plen]
                    prompt_outputs[b_idx][0] = h_valid.detach().to(torch.float32).cpu().numpy()

            # Step 2: Prepare Rotary Embeddings & Position IDs if applicable
            pos_embs = None
            text_pos_ids = None
            if self.rotary_emb is not None:
                pos_ids = torch.arange(seq_len, device=h.device).view(1, 1, -1).expand(4, batch_size, -1)
                text_pos_ids = pos_ids[0]
                pos_ids_rotary = pos_ids[1:]
                pos_embs = self.rotary_emb(h, pos_ids_rotary)

            # Step 3: Prepare Attention Masks
            causal_mask, linear_attn_mask = self._prepare_masks(h, t_mask, text_pos_ids)

            # Step 4: Stream Layer-by-Layer through VRAM
            for l_idx in range(num_layers):
                layer = self.layers[l_idx]
                layer_type = getattr(layer, "layer_type", None)
                if layer_type is None and self.config is not None:
                    layer_types = getattr(self.config, "layer_types", None) or getattr(getattr(self.config, "text_config", None), "layer_types", None)
                    if layer_types is not None and l_idx < len(layer_types):
                        layer_type = layer_types[l_idx]

                # Move active layer to VRAM
                try:
                    layer_vram = layer.to(exec_dev)
                    h_vram = h.to(exec_dev)
                except BaseException as err:
                    print(f"[PipelinedLayerStreamer] VRAM transfer failed for layer {l_idx} ({err}); falling back to CPU.")
                    exec_dev = torch.device("cpu")
                    exec_dev_type = "cpu"
                    layer_vram = layer.to(exec_dev)
                    h_vram = h.to(exec_dev)

                # Prepare layer inputs on execution device
                layer_kwargs: dict[str, Any] = {}
                if pos_embs is not None:
                    if isinstance(pos_embs, (tuple, list)):
                        layer_kwargs["position_embeddings"] = (pos_embs[0].to(exec_dev), pos_embs[1].to(exec_dev))
                    else:
                        layer_kwargs["position_embeddings"] = pos_embs.to(exec_dev)
                if text_pos_ids is not None:
                    layer_kwargs["position_ids"] = text_pos_ids.to(exec_dev)

                # Select appropriate attention mask
                if layer_type == "linear_attention":
                    layer_mask = linear_attn_mask.to(exec_dev) if linear_attn_mask is not None else None
                else:
                    layer_mask = causal_mask.to(exec_dev) if causal_mask is not None else None
                layer_kwargs["attention_mask"] = layer_mask

                # Forward pass through single layer in VRAM
                try:
                    h_next = layer_vram(h_vram, **layer_kwargs)
                except TypeError:
                    # Generic fallback for synthetic or custom layer signatures
                    try:
                        h_next = layer_vram(h_vram, attention_mask=layer_mask)
                    except TypeError:
                        h_next = layer_vram(h_vram)

                if isinstance(h_next, (tuple, list)):
                    h_next = h_next[0]

                if not torch.all(torch.isfinite(h_next)) and exec_dev_type != "cpu":
                    layer_cpu = layer.to("cpu")
                    h_cpu_in = h.to("cpu")
                    layer_kwargs_cpu = {}
                    for k, v in layer_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            layer_kwargs_cpu[k] = v.to("cpu")
                        elif isinstance(v, tuple):
                            layer_kwargs_cpu[k] = tuple(x.to("cpu") if isinstance(x, torch.Tensor) else x for x in v)
                        else:
                            layer_kwargs_cpu[k] = v
                    try:
                        h_next = layer_cpu(h_cpu_in, **layer_kwargs_cpu)
                    except TypeError:
                        h_next = layer_cpu(h_cpu_in)
                    if isinstance(h_next, (tuple, list)):
                        h_next = h_next[0]

                # Offload layer back to CPU to instantly free VRAM
                h = h_next.cpu()
                layer.to("cpu")

                # Record unpadded layer output
                target_idx = l_idx + 1
                if target_set is None or target_idx in target_set:
                    for b_idx, plen in enumerate(unpadded_lens):
                        if padding_side == "left":
                            h_valid = h[b_idx, -plen:]
                        else:
                            h_valid = h[b_idx, :plen]
                        prompt_outputs[b_idx][target_idx] = h_valid.detach().to(torch.float32).numpy()

            # Step 5: Final RMSNorm if present and requested
            final_layer_idx = num_layers
            if self.norm is not None and (target_set is None or final_layer_idx in target_set):
                h_norm = self.norm(h.to(self.norm.weight.device)).cpu()
                for b_idx, plen in enumerate(unpadded_lens):
                    if padding_side == "left":
                        h_valid = h_norm[b_idx, -plen:]
                    else:
                        h_valid = h_norm[b_idx, :plen]
                    prompt_outputs[b_idx][final_layer_idx] = h_valid.detach().to(torch.float32).numpy()

        return StreamOutput(prompt_outputs)

    def collect_hidden_states(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int = 4,
        target_layers: Sequence[int] | None = None,
        device: str | torch.device | None = None,
    ) -> list[dict[int, np.ndarray]]:
        """Extract hidden states for a list of prompts via Layer-Outer VRAM streaming.

        Optimized Layer-Outer Streaming (1 layer at a time):
        Instead of re-loading all layers over PCIe for every prompt batch (which causes
        N_batches * N_layers PCIe transfers), each layer is loaded into VRAM ONCE,
        all prompt batches are passed through it in VRAM, and then the layer is offloaded.
        This provides a ~4x speedup on PCIe-constrained setups (e.g. Radeon RX 580 DirectML)
        while maintaining an ultra-safe minimal VRAM footprint (~172 MB in FP16 / ~345 MB in FP32).

        Drop-in replacement for `scripts.run_qwen35_transfer.collect_hidden_states`.
        """
        prompt_list = list(prompts)
        if not prompt_list:
            return []

        bs = max(1, int(batch_size))
        if len(prompt_list) <= bs:
            stream_out = self.stream_forward(
                prompt_list,
                target_layers=target_layers,
                device=device,
            )
            return list(stream_out)

        if device is not None:
            exec_dev, exec_dev_type = resolve_execution_device(device)
        else:
            exec_dev, exec_dev_type = self.target_device, self.device_type

        target_set = set(target_layers) if target_layers is not None else None
        num_layers = self.num_layers
        padding_side = "right"
        if self.tokenizer is not None and getattr(self.tokenizer, "padding_side", None) is not None:
            padding_side = self.tokenizer.padding_side

        # Pre-process all batches: tokenization, initial embeddings, position embeddings, and masks
        batch_contexts: list[dict[str, Any]] = []
        prompt_outputs: list[dict[int, np.ndarray]] = [{} for _ in range(len(prompt_list))]

        with torch.no_grad():
            for i in range(0, len(prompt_list), bs):
                chunk_prompts = prompt_list[i : i + bs]
                t_ids, t_mask, unpadded_lens = self._prepare_inputs(chunk_prompts)
                b_size, seq_len = t_ids.shape

                # Step 1: Initial Embedding on host CPU
                if self.embed_tokens is not None:
                    h = self.embed_tokens(t_ids.to(self.embed_tokens.weight.device)).cpu()
                else:
                    h = torch.zeros((b_size, seq_len, 32), dtype=self.dtype)

                # Store Layer 0 if requested
                if target_set is None or 0 in target_set:
                    for local_idx, plen in enumerate(unpadded_lens):
                        p_idx = i + local_idx
                        if padding_side == "left":
                            h_valid = h[local_idx, -plen:]
                        else:
                            h_valid = h[local_idx, :plen]
                        prompt_outputs[p_idx][0] = h_valid.detach().to(torch.float32).numpy()

                # Step 2: Prepare Rotary Embeddings & Position IDs
                pos_embs = None
                text_pos_ids = None
                if self.rotary_emb is not None:
                    pos_ids = torch.arange(seq_len, device=h.device).view(1, 1, -1).expand(4, b_size, -1)
                    text_pos_ids = pos_ids[0]
                    pos_ids_rotary = pos_ids[1:]
                    pos_embs = self.rotary_emb(h, pos_ids_rotary)

                # Step 3: Prepare Attention Masks
                causal_mask, linear_attn_mask = self._prepare_masks(h, t_mask, text_pos_ids)

                batch_contexts.append({
                    "start_idx": i,
                    "h": h,
                    "unpadded_lens": unpadded_lens,
                    "pos_embs": pos_embs,
                    "text_pos_ids": text_pos_ids,
                    "causal_mask": causal_mask,
                    "linear_attn_mask": linear_attn_mask,
                })

            # Step 4: Stream Layer-by-Layer through VRAM (Layer-Outer loop)
            for l_idx in range(num_layers):
                layer = self.layers[l_idx]
                layer_type = getattr(layer, "layer_type", None)
                if layer_type is None and self.config is not None:
                    layer_types = getattr(self.config, "layer_types", None) or getattr(getattr(self.config, "text_config", None), "layer_types", None)
                    if layer_types is not None and l_idx < len(layer_types):
                        layer_type = layer_types[l_idx]

                # Move active layer to VRAM ONCE
                try:
                    layer_vram = layer.to(exec_dev)
                except BaseException as err:
                    print(f"[PipelinedLayerStreamer] VRAM transfer failed for layer {l_idx} ({err}); falling back to CPU.")
                    exec_dev = torch.device("cpu")
                    exec_dev_type = "cpu"
                    layer_vram = layer.to(exec_dev)

                target_idx = l_idx + 1
                should_record = target_set is None or target_idx in target_set

                # Process all batches through this active layer in VRAM
                for ctx in batch_contexts:
                    h_vram = ctx["h"].to(exec_dev)
                    layer_kwargs: dict[str, Any] = {}

                    if ctx["pos_embs"] is not None:
                        if isinstance(ctx["pos_embs"], (tuple, list)):
                            layer_kwargs["position_embeddings"] = (
                                ctx["pos_embs"][0].to(exec_dev),
                                ctx["pos_embs"][1].to(exec_dev),
                            )
                        else:
                            layer_kwargs["position_embeddings"] = ctx["pos_embs"].to(exec_dev)
                    if ctx["text_pos_ids"] is not None:
                        layer_kwargs["position_ids"] = ctx["text_pos_ids"].to(exec_dev)

                    if layer_type == "linear_attention":
                        layer_mask = ctx["linear_attn_mask"].to(exec_dev) if ctx["linear_attn_mask"] is not None else None
                    else:
                        layer_mask = ctx["causal_mask"].to(exec_dev) if ctx["causal_mask"] is not None else None
                    layer_kwargs["attention_mask"] = layer_mask

                    try:
                        h_next = layer_vram(h_vram, **layer_kwargs)
                    except TypeError:
                        try:
                            h_next = layer_vram(h_vram, attention_mask=layer_mask)
                        except TypeError:
                            h_next = layer_vram(h_vram)

                    if isinstance(h_next, (tuple, list)):
                        h_next = h_next[0]

                    if not torch.all(torch.isfinite(h_next)) and exec_dev_type != "cpu":
                        layer_cpu = layer.to("cpu")
                        h_cpu_in = ctx["h"].to("cpu")
                        layer_kwargs_cpu = {}
                        for k, v in layer_kwargs.items():
                            if isinstance(v, torch.Tensor):
                                layer_kwargs_cpu[k] = v.to("cpu")
                            elif isinstance(v, tuple):
                                layer_kwargs_cpu[k] = tuple(x.to("cpu") if isinstance(x, torch.Tensor) else x for x in v)
                            else:
                                layer_kwargs_cpu[k] = v
                        try:
                            h_next = layer_cpu(h_cpu_in, **layer_kwargs_cpu)
                        except TypeError:
                            h_next = layer_cpu(h_cpu_in)
                        if isinstance(h_next, (tuple, list)):
                            h_next = h_next[0]
                        # Restore active layer back to VRAM for remaining batches
                        layer.to(exec_dev)

                    # Offload hidden state back to CPU
                    h_cpu = h_next.cpu()
                    ctx["h"] = h_cpu

                    if should_record:
                        start_i = ctx["start_idx"]
                        for local_idx, plen in enumerate(ctx["unpadded_lens"]):
                            p_idx = start_i + local_idx
                            if padding_side == "left":
                                h_valid = h_cpu[local_idx, -plen:]
                            else:
                                h_valid = h_cpu[local_idx, :plen]
                            prompt_outputs[p_idx][target_idx] = h_valid.detach().to(torch.float32).numpy()

                # Offload layer back to CPU instantly to free VRAM
                layer.to("cpu")
                del layer_vram

            # Step 5: Final RMSNorm if present and requested
            final_layer_idx = num_layers
            if self.norm is not None and (target_set is None or final_layer_idx in target_set):
                try:
                    norm_dev = self.norm.to(exec_dev)
                except BaseException:
                    norm_dev = self.norm.to("cpu")

                for ctx in batch_contexts:
                    h_dev = ctx["h"].to(norm_dev.weight.device)
                    h_norm = norm_dev(h_dev).cpu()
                    start_i = ctx["start_idx"]
                    for local_idx, plen in enumerate(ctx["unpadded_lens"]):
                        p_idx = start_i + local_idx
                        if padding_side == "left":
                            h_valid = h_norm[local_idx, -plen:]
                        else:
                            h_valid = h_norm[local_idx, :plen]
                        prompt_outputs[p_idx][final_layer_idx] = h_valid.detach().to(torch.float32).numpy()

                self.norm.to("cpu")
                del norm_dev

        return prompt_outputs


def stream_hidden_states(
    streamer_or_model: PipelinedLayerStreamer | nn.Module | str | Path,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: str | torch.device = "auto",
    batch_size: int = 4,
    target_layers: Sequence[int] | None = None,
) -> list[dict[int, np.ndarray]]:
    """Convenience helper to collect hidden states via pipelined layer streaming.

    Args:
        streamer_or_model: PipelinedLayerStreamer instance, model path, or nn.Module.
        tokenizer: Pre-initialized tokenizer.
        prompts: Sequence of input text prompts.
        device: Execution device ("auto", "dml", "cuda", "cpu").
        batch_size: Batch size for forward chunks.
        target_layers: Optional sequence of layer indices.

    Returns:
        List of dicts mapping layer_idx -> numpy array of shape (seq_len, hidden_size).
    """
    if isinstance(streamer_or_model, PipelinedLayerStreamer):
        streamer = streamer_or_model
    else:
        streamer = PipelinedLayerStreamer(
            streamer_or_model,
            device=device,
            tokenizer=tokenizer,
        )

    return streamer.collect_hidden_states(
        prompts,
        batch_size=batch_size,
        target_layers=target_layers,
        device=device,
    )
