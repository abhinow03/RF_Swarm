"""Builds evaluation/demo_webdata.json -- the pre-computed Act 1 + Act 2
payload the web interface (scripts/demo_web.py) serves without touching the
GPU at demo time. Re-run this whenever config/demo_config.ACTIVE_ADAPTER
changes (e.g. swapping v5-a -> v5a2) to refresh the cache; the web server
itself never regenerates it.

Act 3 stays live in the web interface (that's the point of it) -- this
script does not touch Act 3 at all.

Usage:
    python scripts/build_demo_webdata.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))

import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402
import demo_act1_contrast as act1  # noqa: E402
import demo_act2_pipeline as act2  # noqa: E402

OUT_PATH = REPO / "evaluation" / "demo_webdata.json"


def main() -> None:
    dc.gpu_free_gib(required_gib=15.0)

    client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
    client.add_adapter("v2", demo_config.V2_ADAPTER)
    client.add_adapter("active", demo_config.ACTIVE_ADAPTER)
    print("base model loaded, v2/active adapters registered")

    act1_data = act1.build(client)
    print(f"Act 1 built: {act1_data['case_name']}")

    act2_data = act2.build(client)
    print(f"Act 2 built: {len(act2_data)} cases")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_adapter": demo_config.ACTIVE_ADAPTER,
        "act1": act1_data,
        "act2": act2_data,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_PATH}")

    del client
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
