#!/usr/bin/env python3
"""Aggregate LLAMA_EP_TELEMETRY_FILE JSONL routing records.

The CUDA hook records the global top-k IDs before local 64/64 remapping.  This
tool intentionally keeps the aggregation independent of llama.cpp so the
resulting telemetry is easy to archive and inspect.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Iterable


EXPERTS = 128
SHARD_SIZE = 64
LAYER_RE = re.compile(r"-(\d+)$")


def route_partition(ids: Iterable[int], n_expert_used: int) -> tuple[list[int], int, int]:
    """Return local IDs and ownership counts for one token's global routes."""
    global_ids = list(ids)
    if len(global_ids) != n_expert_used:
        raise ValueError("route slot count does not match n_expert_used")
    if any(expert < 0 or expert >= EXPERTS for expert in global_ids):
        raise ValueError("global expert ID outside 0..127")
    local = [expert if expert < SHARD_SIZE else expert - SHARD_SIZE for expert in global_ids]
    gpu0 = sum(expert < SHARD_SIZE for expert in global_ids)
    return local, gpu0, n_expert_used - gpu0


def decode_local_route(expert: int, shard: int) -> int:
    """Return the decode-kernel local ID, or -1 for a remote route.

    The CUDA MMVQ fast path uses -1 as a non-owning-slot sentinel. Keeping
    this small contract in the telemetry tool makes the fixed 64/64 mapping
    independently testable without requiring two GPUs.
    """
    if expert < 0 or expert >= EXPERTS:
        raise ValueError("global expert ID outside 0..127")
    if shard not in (0, 1):
        raise ValueError("EP shard must be 0 or 1")
    owner = expert // SHARD_SIZE
    return expert % SHARD_SIZE if owner == shard else -1


def aggregate(lines: Iterable[str]) -> dict:
    layers: dict[str, dict] = {}
    co_selection: collections.Counter[tuple[int, int]] = collections.Counter()
    total_counts = [0] * EXPERTS
    total_gpu0 = total_gpu1 = 0
    split_hist = collections.Counter()
    records = 0

    for raw in lines:
        if not raw.strip():
            continue
        record = json.loads(raw)
        ids = [int(value) for value in record["ids"]]
        n_used = int(record["n_expert_used"])
        n_tokens = int(record["n_tokens"])
        if len(ids) != n_used * n_tokens:
            raise ValueError("telemetry record has an unexpected flattened shape")
        match = LAYER_RE.search(record["tensor"])
        layer = match.group(1) if match else record["tensor"]
        bucket = layers.setdefault(layer, {
            "selection_count": [0] * EXPERTS,
            "gpu0_selections": 0,
            "gpu1_selections": 0,
            "ownership_split": {f"{i}/{n_used-i}": 0 for i in range(n_used + 1)},
            "records": 0,
        })
        bucket["records"] += 1
        records += 1
        for token in range(n_tokens):
            token_ids = ids[token * n_used:(token + 1) * n_used]
            _, gpu0, gpu1 = route_partition(token_ids, n_used)
            split_hist[f"{gpu0}/{gpu1}"] += 1
            bucket["ownership_split"][f"{gpu0}/{gpu1}"] += 1
            bucket["gpu0_selections"] += gpu0
            bucket["gpu1_selections"] += gpu1
            total_gpu0 += gpu0
            total_gpu1 += gpu1
            for expert in token_ids:
                bucket["selection_count"][expert] += 1
                total_counts[expert] += 1
            for left in range(n_used):
                for right in range(left + 1, n_used):
                    pair = tuple(sorted((token_ids[left], token_ids[right])))
                    co_selection[pair] += 1

    hottest = sorted(enumerate(total_counts), key=lambda item: (-item[1], item[0]))
    return {
        "experts": EXPERTS,
        "shard_size": SHARD_SIZE,
        "records": records,
        "total_gpu0_selections": total_gpu0,
        "total_gpu1_selections": total_gpu1,
        "gpu0_fraction": total_gpu0 / (total_gpu0 + total_gpu1) if total_gpu0 + total_gpu1 else 0.0,
        "gpu1_fraction": total_gpu1 / (total_gpu0 + total_gpu1) if total_gpu0 + total_gpu1 else 0.0,
        "ownership_split": dict(sorted(split_hist.items())),
        "selection_count": total_counts,
        "hottest_experts": [{"expert": expert, "count": count} for expert, count in hottest if count],
        "co_selection": {f"{left},{right}": count for (left, right), count in co_selection.most_common()},
        "layers": {layer: layers[layer] for layer in sorted(layers, key=lambda value: int(value) if value.isdigit() else value)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = aggregate(args.input.read_text().splitlines())
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
