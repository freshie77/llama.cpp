import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "ep_telemetry.py"
SPEC = importlib.util.spec_from_file_location("ep_telemetry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_global_to_local_mapping_and_ownership():
    local, gpu0, gpu1 = MODULE.route_partition([0, 63, 64, 127, 5, 70, 12, 99], 8)
    assert local == [0, 63, 0, 63, 5, 6, 12, 35]
    assert (gpu0, gpu1) == (4, 4)


def test_telemetry_preserves_duplicate_local_slots():
    # Routes from different global shards may intentionally map to the same
    # local ID after each branch masks the unowned route.  The telemetry must
    # count the original global routes, not collapse them.
    summary = MODULE.aggregate([json.dumps({
        "tensor": "ffn_moe_ep_route_ids-7",
        "n_expert_used": 8,
        "n_tokens": 2,
        "ids": [0, 1, 2, 3, 64, 65, 66, 67, 64, 65, 66, 67, 0, 1, 2, 3],
    })])
    assert summary["total_gpu0_selections"] == 8
    assert summary["total_gpu1_selections"] == 8
    assert summary["ownership_split"] == {"4/4": 2}
    assert summary["selection_count"][0] == 2
    assert summary["selection_count"][64] == 2
    assert summary["layers"]["7"]["records"] == 1


def test_invalid_route_fails_closed():
    try:
        MODULE.route_partition([0, 128], 2)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("invalid expert ID was accepted")
