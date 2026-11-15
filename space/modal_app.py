"""Permanent web deployment of the Pneumodetect Gradio app on Modal (https://modal.com).

    modal deploy modal_app.py          (from this folder; prints the permanent https://...modal.run URL)

PNEUMODETECT_ASSETS must point to a local folder with models/*.keras (outputs of the Kaggle notebook
keonmei/pneumodetect) and examples/*.jpeg. The container scales to zero when idle, so the first request after a
pause takes ~30-60 s (TensorFlow start-up and model loading); after that one prediction takes ~1 s on CPU.
"""
import os
from pathlib import Path

import modal

ASSETS = Path(os.environ.get("PNEUMODETECT_ASSETS", Path(__file__).parent))
HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tensorflow==2.21.0", "keras==3.13.2", "gradio==6.32.0", "fastapi[standard]",
                 "matplotlib", "pillow", "numpy")
    .env({"PNEUMODETECT_ASSETS": "/assets", "TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_dir(ASSETS / "models", "/assets/models")
    .add_local_dir(ASSETS / "examples", "/assets/examples")
    .add_local_file(HERE / "app.py", "/root/pneumodetect_app.py")
)

app = modal.App("pneumodetect", image=image)


@app.function(cpu=2, memory=4096, max_containers=1, scaledown_window=15 * 60, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    # One container (Gradio keeps per-session state), several requests at a time, sleeps after 15 idle minutes.
    import sys

    sys.path.insert(0, "/root")
    import pneumodetect_app

    return pneumodetect_app.create_app()          # Gradio UI (always in English) mounted on a FastAPI app
