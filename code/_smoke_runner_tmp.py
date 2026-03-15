import os
import sys
import json
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = [
    "code/chapter-4/02_containers.py",
    "code/chapter-5/01_loss_function.py",
    "code/chapter-6/04_grad_cam.py",
    "code/chapter-7/06_albumentations-demo.py",
    "code/chapter-8/08_image_retrieval/00_lsh_demo.py",
    "code/chapter-9/a_rnn_lstm/datasets/aclImdb_dataset.py",
    "code/chapter-10/plot_gpu_usage.py",
    "code/chapter-11/00_onnx_graph.py",
    "code/chapter-11/02_resnet_inference.py",
    "code/chapter-12/03_get_resnet50_wts.py",
]

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
results = []

for s in SCRIPTS:
    path = os.path.join(ROOT, s)
    if not os.path.exists(path):
        results.append({"script": s, "status": "FAIL", "reason": "missing_script"})
        continue
    try:
        cp = subprocess.run(
            [sys.executable, path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        if cp.returncode == 0:
            results.append({"script": s, "status": "PASS"})
        else:
            lines = ((cp.stderr or "") + "\n" + (cp.stdout or "")).strip().splitlines()
            reason = lines[-1] if lines else "unknown_error"
            results.append({"script": s, "status": "FAIL", "reason": reason})
    except subprocess.TimeoutExpired:
        results.append({"script": s, "status": "FAIL", "reason": "timeout"})

print(json.dumps(results, ensure_ascii=False, indent=2))
