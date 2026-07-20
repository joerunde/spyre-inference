# Layer-bisection harness

Localize where a model's forward pass diverges on Spyre from a gold reference, by
diffing per-layer (and per-submodule) activations. Reusable for any decoder-LM
Spyre bring-up: when a model runs but produces degenerate/wrong output and every
component passes in isolation, this finds the offending layer and sub-op.

## Scripts

| Script | Role |
|---|---|
| `ref_hf.py` | Gold reference — runs the model in HuggingFace on CPU (fp32), captures per-layer residual stream (`output_hidden_states`) + submodule outputs (hooks). |
| `capture_vllm.py` | System under test — runs the model under vLLM (spyre plugin active), captures the same via `collective_rpc` forward hooks. |
| `diff.py` | Per-module cosine + max-abs diff; prints the first diverging module. |

Activations are keyed with canonical names so the two sides align despite
differing qualified names: `L{i}` = residual stream after decoder layer `i`,
`L{i}.<submodule>` = a submodule output within that layer.

## Usage

```bash
cd tests/tools/layer_bisect
M=google/gemma-3-1b-it

# 1. gold reference (CPU, no accelerator needed)
HF_TOKEN=$HF_TOKEN uv run --no-sync python ref_hf.py --model $M --out /tmp/ref.pt

# 2. Spyre capture (exclusive card access; needs insecure serialization for the RPC callable)
VLLM_ALLOW_INSECURE_SERIALIZATION=1 HF_TOKEN=$HF_TOKEN \
    uv run python capture_vllm.py --model $M --out /tmp/spyre.pt

# 3. diff
uv run --no-sync python diff.py /tmp/ref.pt /tmp/spyre.pt
```

Keep the same `--prompt` on both sides (default identical). Compare `token_ids`
printed by `ref_hf.py` against the vLLM tokenization if in doubt.

## Reading the output

- `L{i}` cosine ~1.0 across all layers → the model matches; the bug is elsewhere
  (sampling, logits, lm_head).
- Cosine drops at layer `K` and stays low → divergence onset is layer `K`. Look at
  the `L{K}.*` submodule rows: the first submodule with low cosine is the culprit.
- **Terminal-layer caveat:** the last decoder layer's reconstructed residual
  stream (`out[0]+out[1]`) can false-positive even when the model is correct.
  Trust the per-submodule cosines and the generated text there.
- **Capture-boundary caveat:** HF and vLLM sometimes split a fused op across
  different module boundaries (e.g. HF `mlp.act_fn` = `gelu(gate)` vs vLLM's fused
  `GeluAndMul` = `gelu(gate)*up`). Such rows show low cosine by construction —
  judge by the module *output* (`L{i}.mlp`), not the internal split.

## Worked example (gemma-3-1b, 2026-07-17)

The harness localized "gemma-3 emits repeated-token garbage" to the MLP: every L0
submodule was cosine 1.0 except `L0.mlp` (0.26). Root cause was `GeluAndMul`
(gelu_pytorch_tanh) lacking a Spyre override, so the fused gate/up tensor was
sliced on-device (memory corruption). Adding `SpyreGeluAndMul` restored coherent
output (all layers back to cosine 1.0). See
`GEMMA3_GENERATION_QUALITY_REPORT.md`.

## Requirements / notes

- Reference is HuggingFace (device-independent), not vLLM-on-CPU: this repo's
  vLLM is built `VLLM_TARGET_DEVICE=empty` and has no platform without the plugin.
- fp32 reference vs fp16 Spyre is fine — cosine is robust to that; correct layers
  read ~0.999+.
- The `.layers.`-based submodule hooks assume a `...layers.N...` naming (Llama /
  Gemma / Qwen / Mistral-style). The per-layer `L{i}` signal comes from
  `output_hidden_states` and is architecture-agnostic.
- Single Spyre card: never run `capture_vllm.py` concurrently with another
  Spyre process.
