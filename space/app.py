"""Pneumodetect web demo.

Loads the two models trained in Pneumodetect.ipynb (binary pneumonia model and Normal / Bacterial / Viral model)
and serves them with Gradio. Same preprocessing and Grad-CAM as in Pneumodetect_App.ipynb.
Course project, not a medical device.

Run locally (CPU is enough, ~1 s per X-ray):
    pip install -r requirements.txt gradio
    set PNEUMODETECT_ASSETS=C:\\path\\to\\folder        (contains models/*.keras and, optionally, examples/*.jpeg)
    set PNEUMODETECT_SHARE=1                           (optional: temporary public https://....gradio.live link)
    python app.py
The two .keras files are the outputs of the Kaggle notebook keonmei/pneumodetect.
"""
import inspect
import os
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


def predict(image):
    if image is None:
        return None, "Upload a frontal chest X-ray.", None
    gray, x = preprocess(image)
    cam, out = grad_cam(BINARY, x)
    p = float(out[0, 0])
    probs3 = MODEL3(x, training=False).numpy()[0].astype(float)
    bacterial = probs3[1] / (probs3[1] + probs3[2])
    return {"Pneumonia": p, "Normal": 1 - p}, report(p, bacterial), overlay(gray, cam[0])


predict(Image.new("L", (IMG_SIZE, IMG_SIZE)))       # warm-up: trace the graphs once at start-up

examples = sorted(str(p) for p in (ASSETS / "examples").glob("*.jpeg"))
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Frontal chest X-ray (PA/AP), JPEG or PNG"),
    outputs=[gr.Label(num_top_classes=2, label="Probability"),
             gr.Textbox(label="Result"),
             gr.Image(label="Grad-CAM: regions that influenced the prediction")],
    examples=[[e] for e in examples] or None,
    title="Pneumodetect — pneumonia detection from chest X-rays",
    description=(
        "ResNet50 fine-tuned on ~108k chest X-rays (NIH ChestX-ray14 with RSNA radiologist labels + Kermany "
        "paediatric X-rays). Returns P(pneumonia), the likely type (bacterial / viral) and a Grad-CAM heat map; "
        "probabilities between 40% and 60% are flagged for manual review.\n\n"
        "**Course project (Machine Learning Algorithms, Astana IT University) — not a medical device, "
        "not a diagnosis.** Example images: Kermany et al. (2018), Chest X-Ray Images (Pneumonia), CC BY 4.0."),
    **({"flagging_mode": "never"} if "flagging_mode" in inspect.signature(gr.Interface).parameters else {}),
)

if __name__ == "__main__":
    # allowed_paths: Gradio only serves files from the working directory unless told otherwise (example X-rays).
    demo.launch(share=os.environ.get("PNEUMODETECT_SHARE") == "1", allowed_paths=[str(ASSETS)])
