# Qwen3-VL fixed expert parallelism (P100 proof of concept)

This isolated branch adds `--split-mode expert` (alias `ep`) for exactly two
CUDA devices and the Qwen3/Qwen3-VL MoE model used on the T7910.  It is not a
fallback: unsupported architectures or device counts fail during model load.

## Placement and execution

Every 128-expert gate/up/down tensor is loaded as two local 64-expert tensors.
Experts `0..63` are resident on CUDA device 0 and experts `64..127` on device
1.  The router still produces global top-8 IDs and the original routing
weights.  Each branch remaps IDs to its local namespace, masks routes it does
not own, evaluates whole local experts with `MUL_MAT_ID`, and the two weighted
partial sums are added.  This is whole-expert sharding; the expert matrices
are not split column-wise or row-wise.

The attention, norms, router, and other non-expert tensors use the normal
llama.cpp device placement selected by `--tensor-split 1,1`.  v1 deliberately
does not add a second tensor-parallel implementation for those shared paths.
The branch dependencies let the existing CUDA backend copy data between the
two P100s as required; no silent CPU or stock tensor-split fallback is used.

`MUL_MAT_ID` normally assumes unique IDs for a top-k set.  EP masks an
unowned route after remapping, so the dummy local ID can repeat.  The CUDA
compaction path therefore retains every route slot instead of dropping a
duplicate.  Its output remains zero-weighted for the branch that does not own
that route.

## Run

```sh
CUDA_VISIBLE_DEVICES=0,1 ./build-p100-ep/bin/llama-server \
  --model /home/j/models/qwen3-vl-30b-a3b-uncensored-q4/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-Q4_K_M.gguf \
  --mmproj /home/j/models/qwen3-vl-30b-a3b-uncensored-q4/mmproj-F16.gguf \
  --n-gpu-layers 999 --split-mode expert --tensor-split 1,1 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16
```

Set `LLAMA_EP_TELEMETRY_FILE=/path/routes.jsonl` to record global routing
IDs.  The hook is off unless this variable is set and intentionally
synchronizes once per recorded routing tensor, so it should not be enabled for
latency measurements unless telemetry is part of the experiment.  Aggregate
it with:

```sh
python3 tools/ep_telemetry.py /path/routes.jsonl --output /path/routes.json
```

The JSON includes per-layer 128-expert selection counts, GPU ownership totals,
the 8/0 through 0/8 token split histogram, and co-selection counts.

## Validation scope

`tests/test-expert-parallel-routing.py` validates the global-to-local mapping,
ownership accounting, duplicate local slots, and fail-closed invalid IDs.
The runtime smoke test must use the same prompt and settings with stock layer
or tensor mode and compare first-token/logit behavior where the selected
llama.cpp API exposes it.  A coherent completion alone is not numerical proof.
