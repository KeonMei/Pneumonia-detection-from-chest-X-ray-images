---
title: Pneumodetect
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Pneumonia detection from chest X-rays (course project)
---

# Pneumodetect

Decision-support prototype that estimates the probability of pneumonia on a frontal chest X-ray, the likely
type (bacterial / viral) and shows a Grad-CAM heat map.

* Model: ResNet50 (ImageNet) fine-tuned on ~108k chest X-rays — NIH ChestX-ray14 with RSNA radiologist labels
  (adults) and Kermany et al. (2018) paediatric X-rays. Code and results:
  https://github.com/KeonMei/pneumonia-detection-from-chest-x-ray-images
* Held-out Kermany test set (624 images): accuracy 0.873, sensitivity 0.977, specificity 0.701, F1 0.906,
  ROC-AUC 0.944. Three-class model (Normal / Bacterial / Viral): macro-F1 0.812.
* Example images: Kermany et al. (2018), *Chest X-Ray Images (Pneumonia)*, CC BY 4.0.

Course project for *Machine Learning Algorithms*, Astana IT University (Asulkhan Aidarkhanov, Ravil Serikuly,
BDA-2401). **Not a medical device; the output is not a diagnosis.**
