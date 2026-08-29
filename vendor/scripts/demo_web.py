"""Web interface for the defense demo -- same data as run_demo.py's terminal
walkthrough, laid out for readability, with buttons to switch between the
3 systems in Act 1's contrast.

Act 1 + Act 2 are served from evaluation/demo_webdata.json (pre-computed by
scripts/build_demo_webdata.py) -- instant, no GPU dependency, zero risk of
an OOM crash mid-presentation on this shared GPU. Act 3 stays live: one
HotSwapClient (config.demo_config.ACTIVE_ADAPTER only, matching
demo_act3_live.py) is loaded once at server startup and reused for every
/api/act3 request, same hot-swap discipline as the terminal demo.

Before serving anything: re-verifies the protected v5-a checkpoint safety
copy and checks free GPU memory, halting rather than starting a server that
would crash on first live request.

Usage:
    python scripts/demo_web.py
    # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402

import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402
import demo_act3_live as act3  # noqa: E402

WEBDATA_PATH = REPO / "evaluation" / "demo_webdata.json"
STATIC_DIR = REPO / "web"

app = Flask(__name__, static_folder=None)

_state = {"client": None, "webdata": None, "checkpoint_ok": None}


def verify_protected_checkpoint() -> dict:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "phase3a_verify_safety_copy.py")],
        capture_output=True, text=True)
    ok = result.returncode == 0
    return {"ok": ok, "stdout": result.stdout, "stderr": result.stderr}


def load_webdata() -> dict:
    if not WEBDATA_PATH.exists():
        raise FileNotFoundError(
            f"{WEBDATA_PATH} not found -- run `python scripts/build_demo_webdata.py` first "
            f"to pre-compute Act 1/2 (requires GPU, run once ahead of the demo)")
    with open(WEBDATA_PATH) as f:
        data = json.load(f)
    if data.get("active_adapter") != demo_config.ACTIVE_ADAPTER:
        print(f"WARNING: cached webdata was built against {data.get('active_adapter')!r} but "
             f"config/demo_config.ACTIVE_ADAPTER is now {demo_config.ACTIVE_ADAPTER!r} -- "
             f"re-run build_demo_webdata.py to refresh the cache before presenting.")
    return data


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/status")
def api_status():
    return jsonify({
        "active_adapter": demo_config.ACTIVE_ADAPTER,
        "checkpoint_ok": _state["checkpoint_ok"],
        "webdata_generated_at_utc": (_state["webdata"] or {}).get("generated_at_utc"),
        "webdata_active_adapter": (_state["webdata"] or {}).get("active_adapter"),
        "formations": BASE_FORMATIONS,
    })


@app.route("/api/act1")
def api_act1():
    return jsonify(_state["webdata"]["act1"])


@app.route("/api/act2")
def api_act2():
    return jsonify(_state["webdata"]["act2"])


@app.route("/api/act3_disclosure")
def api_act3_disclosure():
    return jsonify({"text": act3.DISCLOSURE})


@app.route("/api/act3", methods=["POST"])
def api_act3():
    body = request.get_json(force=True, silent=True) or {}
    form_a, form_b = body.get("form_a"), body.get("form_b")
    if not form_a or not form_b:
        return jsonify({"error": "form_a and form_b are required"}), 400
    data = act3.run_scenario(_state["client"], form_a, form_b)
    return jsonify(data)


def main() -> None:
    print(f"ACTIVE_ADAPTER = {demo_config.ACTIVE_ADAPTER}")

    check = verify_protected_checkpoint()
    print(check["stdout"])
    if not check["ok"]:
        print(check["stderr"])
        raise SystemExit("HALTING: protected checkpoint safety copy failed verification.")
    _state["checkpoint_ok"] = True

    _state["webdata"] = load_webdata()
    print(f"Act 1/2 webdata loaded (generated {_state['webdata']['generated_at_utc']})")

    free_gib = dc.gpu_free_gib(required_gib=15.0)
    print(f"free GPU memory: {free_gib:.1f} GiB -- loading live model for Act 3")

    client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
    client.add_adapter("active", demo_config.ACTIVE_ADAPTER)
    _state["client"] = client
    print("Act 3 live client ready")

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
