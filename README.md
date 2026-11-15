# Pneumonia-detection-from-chest-X-ray-images

**Pneumodetect** — course project (Machine Learning Algorithms, Astana IT University, BDA-2401):
a ResNet50 transfer-learning model that estimates the probability of pneumonia on a frontal chest X-ray,
the likely type (bacterial / viral) and shows a Grad-CAM heat map. Not a medical device.

**Live demo:** https://keonmei--pneumodetect-web.us-east.modal.run
(the app sleeps when idle — the first request may take up to a minute).

| File | Contents |
|---|---|
| `Pneumodetect.ipynb` | Data pipeline (~108k X-rays: NIH ChestX-ray14 + RSNA labels + Kermany), baselines, CNN, Grad-CAM, 3-class model - executed, with all results |
| `Pneumodetect_App.ipynb` | Gradio demo notebook: loads the saved models, examples on held-out test X-rays |
| `space/app.py` | The same Gradio app as a standalone script |
| `space/modal_app.py` | Deployment of the app on Modal (`modal deploy modal_app.py`) |

Run instructions are in the first (README) cell of each notebook.
