"""Pneumodetect web demo.

Loads the two models trained in Pneumodetect.ipynb (binary pneumonia model and Normal / Bacterial / Viral model)
and serves them with Gradio. Same preprocessing and Grad-CAM as in Pneumodetect_App.ipynb.
Course project, not a medical device.

Run locally (CPU is enough, ~1 s per X-ray), then open http://127.0.0.1:7860:
    pip install -r requirements.txt gradio
    set PNEUMODETECT_ASSETS=C:\\path\\to\\folder        (contains models/*.keras and, optionally, examples/*.jpeg)
    python app.py
The two .keras files are the outputs of the Kaggle notebook keonmei/pneumodetect. The permanent web version runs
on Modal (modal_app.py).
"""
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224                       # the models were trained on 224x224 inputs
UNCERTAIN = (0.40, 0.60)             # probabilities in this band are flagged for manual review
DECISION_THRESHOLD = 0.5
ASSETS = Path(os.environ.get("PNEUMODETECT_ASSETS", Path(__file__).parent))   # folder with models/ and examples/

BINARY = tf.keras.models.load_model(ASSETS / "models" / "pneumodetect_resnet50_best.keras")
MODEL3 = tf.keras.models.load_model(ASSETS / "models" / "pneumodetect_resnet50_3class_best.keras")


def preprocess(image):
    # Any PIL image / numpy array -> (224x224 uint8 grayscale, float32 model input of shape (1, 224, 224, 3)).
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    gray = image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR, reducing_gap=2.0)
    gray = np.asarray(gray, dtype=np.uint8)
    return gray, np.repeat(gray[None, ..., None], 3, axis=-1).astype(np.float32)


def split_model(model):
    # input -> augmentation -> preprocessing -> backbone -> head; returns (before, backbone, head).
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model) and l.name != "augmentation")
    i = model.layers.index(base)
    before = [l for l in model.layers[1:i] if l.name != "augmentation"]      # augmentation is off at inference
    return before, base, model.layers[i + 1:]


def grad_cam(model, images):
    # Grad-CAM of the class logit (evaluated in float32 so that confident predictions do not give empty maps).
    before, base, head = split_model(model)
    dense = head[-1]
    kernel, bias = tf.cast(dense.kernel, tf.float32), tf.cast(dense.bias, tf.float32)
    x = tf.convert_to_tensor(images, tf.float32)
    for layer in before:
        x = layer(x, training=False)
    fmap = tf.cast(base(x, training=False), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(fmap)
        logits = tf.matmul(tf.reduce_mean(fmap, axis=(1, 2)), kernel) + bias
        target, out = logits[:, 0], tf.sigmoid(logits)
    grads = tape.gradient(target, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE), method="bilinear")[..., 0]
    return cam.numpy(), out.numpy()


def overlay(gray, cam, alpha=0.35):
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    base = np.repeat(gray[..., None] / 255.0, 3, axis=-1)
    return np.clip(255 * ((1 - alpha) * base + alpha * heat), 0, 255).astype(np.uint8)


def report(p, bacterial):
    text = f"Pneumonia: {100 * p:.1f}% / Normal: {100 * (1 - p):.1f}%."
    if p >= DECISION_THRESHOLD:
        text += f" Predicted type: Bacterial ({100 * bacterial:.0f}%), Viral ({100 * (1 - bacterial):.0f}%)."
    if UNCERTAIN[0] <= p <= UNCERTAIN[1]:
        text += " Low confidence – manual review recommended."
    elif p < DECISION_THRESHOLD:
        text += " No pneumonia detected."
    return text


def analyse(image):
    gray, x = preprocess(image)
    cam, out = grad_cam(BINARY, x)
    p = float(out[0, 0])                                     # P(pneumonia), binary model
    probs3 = MODEL3(x, training=False).numpy()[0].astype(float)
    bacterial = float(probs3[1] / (probs3[1] + probs3[2]))  # type, given that it is pneumonia
    return p, bacterial, overlay(gray, cam[0])


# ---------------------------------------------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------------------------------------------
def bar(label, value, tone):
    return (f'<div class="pd-bar"><div class="pd-bar-head"><span>{label}</span><b>{100 * value:.1f}%</b></div>'
            f'<div class="pd-bar-track"><div class="pd-bar-fill {tone}" style="width:{100 * value:.1f}%"></div>'
            f'</div></div>')


def verdict_html(p, bacterial):
    if UNCERTAIN[0] <= p <= UNCERTAIN[1]:
        tone, icon, title, note = "warn", "⚠", "Low confidence", "Manual review recommended."
    elif p >= DECISION_THRESHOLD:
        kind = "bacterial" if bacterial >= 0.5 else "viral"
        tone, icon, title, note = "bad", "●", "Pneumonia likely", f"Most likely {kind} pneumonia."
    else:
        tone, icon, title, note = "good", "✓", "No pneumonia detected", "The lungs look clear to the model."
    html = (f'<div class="pd-card"><div class="pd-verdict {tone}"><span class="pd-icon">{icon}</span>'
            f'<div><div class="pd-title">{title}</div><div class="pd-note">{note}</div></div></div>'
            f'<div class="pd-section">Probability</div>'
            + bar("Pneumonia", p, "bad") + bar("Normal", 1 - p, "good"))
    if p >= DECISION_THRESHOLD:
        html += ('<div class="pd-section">Pneumonia type</div>'
                 + bar("Bacterial", bacterial, "accent") + bar("Viral", 1 - bacterial, "accent2"))
    html += f'<div class="pd-report">{report(p, bacterial)}</div></div>'
    return html


PLACEHOLDER = ('<div class="pd-card pd-empty"><div class="pd-empty-icon">🫁</div>'
               '<div class="pd-title">No X-ray analysed yet</div>'
               '<div class="pd-note">Upload a frontal chest X-ray or pick one of the test images, '
               'then press <b>Analyze X-ray</b>.</div></div>')


def predict(image):
    if image is None:
        return PLACEHOLDER, None
    p, bacterial, heat = analyse(image)
    return verdict_html(p, bacterial), heat


HEADER = """
<div class="pd-hero">
  <div class="pd-logo">🫁</div>
  <div>
    <h1>Pneumodetect</h1>
    <p>Pneumonia detection from chest X-rays with a fine-tuned ResNet50 and Grad-CAM explanations</p>
    <div class="pd-chips">
      <span>ResNet50 · transfer learning</span><span>~108k training X-rays</span>
      <span>Bacterial / viral type</span><span>Grad-CAM heat map</span>
    </div>
  </div>
</div>
"""

ABOUT = """
**Data.** NIH ChestX-ray14 (adults) with RSNA radiologist labels and the Kermany et al. (2018) paediatric dataset —
107,858 frontal X-rays, split by patient so that no person appears in two splits.

**Model.** ResNet50 pre-trained on ImageNet, new sigmoid head, two-stage training (head, then the last
convolutional block), class weights, augmentation. A second ResNet50 head separates bacterial and viral pneumonia.

| Held-out Kermany test set (624 X-rays) | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Best classical baseline (ResNet50 features + LogReg) | 0.740 | 1.000 | 0.308 | 0.828 | 0.916 |
| **Pneumodetect (fine-tuned ResNet50)** | **0.873** | **0.977** | **0.701** | **0.906** | **0.944** |

Three-class model (Normal / Bacterial / Viral): macro-F1 0.812. Grad-CAM peak falls inside a radiologist's box
in 58% of NIH test X-rays (chance: 13%).

**Reading the result.** Probabilities between 40% and 60% are flagged as *low confidence*. The model is most
reliable on paediatric X-rays like the examples; on normal children's X-rays it produces some false alarms.
"""

FOOTER = """
<div class="pd-footer">
  <span><b>Pneumodetect</b> · Machine Learning Algorithms course project · Astana IT University · BDA-2401</span>
  <span>Asulkhan Aidarkhanov · Ravil Serikuly</span>
  <span><a href="https://github.com/KeonMei/pneumonia-detection-from-chest-x-ray-images" target="_blank">GitHub</a>
        · Built with Gradio & TensorFlow</span>
  <span class="pd-disclaimer">Not a medical device. The output is not a diagnosis.
        Example images: Kermany et al. (2018), CC BY 4.0.</span>
</div>
"""

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.pd-hero { display: flex; gap: 18px; align-items: center; padding: 26px 28px; border-radius: 18px;
  background: linear-gradient(135deg, #312e81 0%, #4338ca 45%, #0ea5e9 100%); color: #fff; margin-bottom: 6px; }
.pd-hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; }
.pd-hero p { margin: 4px 0 12px; opacity: .9; font-size: 1.02rem; color: #fff; }
.pd-logo { font-size: 3rem; background: rgba(255,255,255,.15); border-radius: 16px; padding: 8px 14px; }
.pd-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.pd-chips span { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
  padding: 4px 12px; font-size: .82rem; color: #fff; }
.pd-card { border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 20px 22px;
  background: var(--block-background-fill); }
.pd-empty { text-align: center; padding: 46px 22px; }
.pd-empty-icon { font-size: 2.6rem; margin-bottom: 8px; }
.pd-verdict { display: flex; gap: 14px; align-items: center; border-radius: 12px; padding: 14px 16px; margin-bottom: 14px; }
.pd-verdict.good { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.45); }
.pd-verdict.bad  { background: rgba(239,68,68,.11);  border: 1px solid rgba(239,68,68,.45); }
.pd-verdict.warn { background: rgba(245,158,11,.13); border: 1px solid rgba(245,158,11,.5); }
.pd-icon { font-size: 1.6rem; width: 2.2rem; text-align: center; }
.good .pd-icon { color: #10b981; } .bad .pd-icon { color: #ef4444; } .warn .pd-icon { color: #f59e0b; }
.pd-title { font-size: 1.25rem; font-weight: 700; }
.pd-note { opacity: .8; font-size: .95rem; }
.pd-section { text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; opacity: .6;
  margin: 14px 0 6px; }
.pd-bar { margin: 6px 0; }
.pd-bar-head { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: 4px; }
.pd-bar-track { height: 9px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden; }
.pd-bar-fill { height: 100%; border-radius: 999px; }
.pd-bar-fill.bad { background: #ef4444; } .pd-bar-fill.good { background: #10b981; }
.pd-bar-fill.accent { background: #6366f1; } .pd-bar-fill.accent2 { background: #0ea5e9; }
.pd-report { margin-top: 14px; font-size: .85rem; opacity: .75; border-top: 1px dashed var(--border-color-primary);
  padding-top: 10px; }
.pd-footer { display: flex; flex-direction: column; align-items: center; gap: 4px; margin-top: 18px; padding: 18px;
  font-size: .85rem; opacity: .75; text-align: center; border-top: 1px solid var(--border-color-primary); }
.pd-footer a { color: var(--color-accent); }
.pd-disclaimer { font-size: .78rem; }
"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])

# theme / css are launch() options in Gradio 6; footer_links=[] hides Gradio's own footer (we draw an English one).
LAUNCH_OPTIONS = {"theme": THEME, "css": CSS, "footer_links": []}

EXAMPLE_NAMES = {"bacteria": "Bacterial pneumonia", "virus": "Viral pneumonia", "NORMAL2-IM-0315": "Normal",
                 "NORMAL2-IM-0351": "Borderline case"}
examples = sorted(str(p) for p in (ASSETS / "examples").glob("*.jpeg"))
example_labels = [next((v for k, v in EXAMPLE_NAMES.items() if k in Path(e).name), Path(e).stem) for e in examples]

predict(Image.new("L", (IMG_SIZE, IMG_SIZE)))       # warm-up: trace the graphs once at start-up

with gr.Blocks(title="Pneumodetect — pneumonia detection from chest X-rays") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            xray = gr.Image(type="pil", label="Chest X-ray (frontal, PA/AP)", height=400,
                            sources=["upload", "clipboard"])
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                run_btn = gr.Button("Analyze X-ray", variant="primary")
        with gr.Column(scale=6):
            verdict = gr.HTML(PLACEHOLDER)
            heatmap = gr.Image(label="Grad-CAM — regions that drove the prediction", height=400, interactive=False)
    if examples:
        gr.Examples(examples=[[e] for e in examples], inputs=xray, outputs=[verdict, heatmap], fn=predict,
                    run_on_click=True, cache_examples=False, example_labels=example_labels,
                    label="Try a held-out test X-ray (Kermany test set)")
    with gr.Accordion("About the model and the results", open=False):
        gr.Markdown(ABOUT)
    gr.HTML(FOOTER)

    run_btn.click(predict, inputs=xray, outputs=[verdict, heatmap])
    clear_btn.click(lambda: (None, PLACEHOLDER, None), outputs=[xray, verdict, heatmap])

# ---------------------------------------------------------------------------------------------------------------
# Serving: Gradio translates its own buttons ("Drop Image Here", ...) into the browser language by loading one
# locale file per language. To keep the whole page in English, every locale file request is answered with the
# English one.
# ---------------------------------------------------------------------------------------------------------------
_GRADIO_ASSETS = Path(gr.__file__).parent / "templates" / "frontend" / "assets"
_LOCALE_FILE = re.compile(r"^[a-z]{2}(-[A-Za-z]{2,4})?-[A-Za-z0-9_-]{8}\.js$")
_LOCALES = {p.name for p in _GRADIO_ASSETS.glob("*.js")
            if _LOCALE_FILE.match(p.name) and "as _name" in p.read_text(encoding="utf-8", errors="ignore")}
_ENGLISH = next((n for n in _LOCALES if n.startswith("en-")), None)


class EnglishOnly:
    """ASGI middleware that serves Gradio's English locale file whatever language the browser asks for."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and _ENGLISH:
            name = scope["path"].rsplit("/", 1)[-1]
            if name in _LOCALES and name != _ENGLISH:
                path = scope["path"][: -len(name)] + _ENGLISH
                scope = {**scope, "path": path, "raw_path": path.encode()}
        await self.app(scope, receive, send)


def create_app():
    from fastapi import FastAPI

    # allowed_paths: Gradio only serves files from the working directory unless told otherwise (example X-rays).
    app = gr.mount_gradio_app(FastAPI(), demo, path="/", allowed_paths=[str(ASSETS)], **LAUNCH_OPTIONS)
    return EnglishOnly(app)


if __name__ == "__main__":
    import uvicorn

    print("Pneumodetect: http://127.0.0.1:7860")
    uvicorn.run(create_app(), host="127.0.0.1", port=7860, log_level="warning")
