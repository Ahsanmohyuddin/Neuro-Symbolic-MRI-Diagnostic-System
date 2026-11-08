[Uploading README.md…]()
# NeuroScan: Alzheimer's MRI Diagnostic System

NeuroScan is a web application that classifies brain MRI scans into four stages of Alzheimer's disease. It uses a ConvNeXt-Small model trained on the OASIS dataset and shows a Grad-CAM heatmap with every prediction so the result can be checked visually.

This is a final year project. It is not a medical device and should not be used for real diagnosis.

## Why this project

Alzheimer's disease makes up 60-80% of dementia cases, and over 55 million people live with dementia worldwide. It is usually diagnosed late, after a lot of brain damage has happened. The normal process (cognitive tests, MRI, specialist review) can take weeks or months and is not available everywhere.

Most deep learning tools for medical images stay as research code and give no explanation for their answers. This project tries to fix that with a model that gives an explanation and a simple web interface that runs in any browser.

## What it does

- Upload an MRI image (JPG or PNG), by browsing or drag and drop
- Predicts one of four classes: Non-Demented, Very Mild, Mild, Moderate Demented
- Shows the confidence and the probability of each class
- Shows a Grad-CAM heatmap on top of the scan
- Survey based on the predicted stage, with a PDF report
- Doctor panel for booking an online session
- Information page about the disease stage
- History that the user can delete
- Images are processed in memory and never saved

## Results

Tested on a held-out 20% test split of OASIS.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-Demented | 0.96 | 0.98 | 0.97 |
| Very Mild Demented | 0.91 | 0.89 | 0.90 |
| Mild Demented | 0.88 | 0.86 | 0.87 |
| Moderate Demented | 0.83 | 0.79 | 0.81 |
| Weighted average | 0.93 | 0.93 | 0.93 |

Overall accuracy is about 93% and balanced accuracy about 88%. AUC-ROC is between 0.95 and 0.99 for all classes. Most mistakes are between neighbouring stages (Very Mild and Mild). Moderate Demented is the weakest class because the dataset has only about 64 original images of it.

The Grad-CAM heatmaps mostly focus on the temporal lobes, hippocampus and entorhinal cortex, which are the areas affected first in Alzheimer's.

## Dataset

OASIS Brain MRI, with classes based on the Clinical Dementia Rating (CDR).

| Class | CDR | Approx. images |
|---|---|---|
| Non-Demented | 0 | 3,200 |
| Very Mild Demented | 0.5 | 2,240 |
| Mild Demented | 1 | 896 |
| Moderate Demented | 2 | 64 |

- Images resized to 256x256, converted to RGB and normalized (mean 0.5, std 0.5)
- Split 60/20/20 for train, validation and test, with seed 42
- Augmentation: horizontal flip, rotation of 20 degrees, color jitter of 0.2
- Minority classes oversampled to 2,000 training images; the test set keeps its original distribution

## Model and training

The model is ConvNeXt-Small with ImageNet-1K weights. The final layer was replaced with a 4-class linear layer (768 to 4) and only feature blocks 8 and 9 were fine-tuned. Everything else stayed frozen.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Loss | Cross-entropy |
| Batch size | 32 |
| Epochs | 20 |
| Input size | 256 x 256 |
| Hardware | Google Colab (NVIDIA T4) |

Training accuracy ended around 96% against about 93% on the test set, which is a small amount of overfitting.

## Grad-CAM

Grad-CAM uses forward and backward hooks on the last convolutional block (`features.7`). The gradients of the predicted class are averaged per channel and used to weight the activations. The result goes through ReLU, is resized to 256x256, colored with the jet colormap and blended over the original image at 40% opacity.

## How it works

The backend (FastAPI) loads the model once when it starts. When an image arrives it is preprocessed, passed through the model, and the Grad-CAM is calculated from the same forward pass. The result is returned as JSON with a base64 image. The frontend is one HTML file that calls the API with fetch and shows the results.

If the model file is missing, the server still starts but returns 503 on predictions. An invalid image returns 400.

### API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check, shows whether the model is loaded |
| POST | `/predict` | Takes an image in the `file` field, returns the prediction |
| GET | `/docs` | Interactive API docs |

The prediction response contains the predicted class, the confidence (0-100), the probabilities of the four classes and the Grad-CAM image as a base64 PNG.

CORS is open to all origins for local development. Change this before deploying.

## Setup

You need Python 3.9 or newer and the trained model file.

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install torch torchvision fastapi uvicorn opencv-python numpy pillow python-multipart
```

Put the model `.pth` file in the same folder as `main.py`, then start the server:

```bash
uvicorn main:app
```

The API runs at http://localhost:8000. Open `index.html` in a browser, choose an MRI image and click Analyze.

For cloud deployment, a Docker image based on a Python image works: install the dependencies, copy the model and `main.py`, expose port 8000 and run uvicorn. Change the API address in `index.html` and use HTTPS.

### Requirements

- Server: 2 CPU cores, 4 GB RAM (8 GB recommended), 2 GB disk. A GPU is optional.
- Client: any modern browser
- Versions used: Python 3.10, PyTorch 2.1.0, torchvision 0.16.0, FastAPI 0.111.0, Uvicorn 0.29.0, OpenCV 4.8.0, NumPy 1.24.0, Pillow 10.0.0

The model itself takes about 1 second per image on a CPU and under 100 ms on a GPU. Grad-CAM and encoding add some time on a CPU.

## Testing

- The model was evaluated on the test split with scikit-learn (classification report, confusion matrix, ROC-AUC)
- The API was tested with Postman and curl for status codes, JSON format and image validity
- End-to-end tests were done manually with images from all four classes

Manual test cases covered a valid upload, a mild case, a non-image file (400), heatmap display, probability bars, missing model (503), drag and drop, and opening the page from `file://`. All passed.

## Limitations

- Trained only on OASIS, so it may not work as well on other scanners or patient groups
- Uses single 2D axial slices, not 3D volumes, and does not accept DICOM, CT or PET
- Does not use patient information such as age or MMSE score
- Few Moderate Demented examples, so that class has the lowest recall
- Grad-CAM has a lower resolution than the input image
- Analyzes one image at a time

## Ethics

NeuroScan is meant to support a doctor, not replace one. Images are not stored. Real clinical use would need to follow rules like HIPAA or GDPR, and the model should be tested on more diverse data to check for bias.

## Future work

- LLM-based explanation of the stage and next steps
- Urdu and English chatbot
- Specialist and doctor listing
- Deployment on a live server with a domain
- 3D MRI and PET analysis
- Faster inference with TorchScript, ONNX or quantization

## References

1. Z. Liu et al., "A ConvNet for the 2020s," CVPR, 2022.
2. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," IJCV, 2020.
3. A. Farooq et al., "A Deep CNN Based Multi-Class Classification of Alzheimer's Disease using MRI," IEEE ISICT, 2017.
4. M. Liu et al., "A multi-model deep convolutional neural network for automatic hippocampus segmentation and classification in Alzheimer's disease," NeuroImage, vol. 208, 2020.
5. D. S. Marcus et al., "Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data," J. Cogn. Neurosci., 2007.
6. World Health Organization, "Dementia Fact Sheet," 2023.
7. E. Hosseini-Asl et al., "Alzheimer's Disease Diagnosis Using a Deeply Supervised Adaptive 3D Convolutional Network," arXiv:1607.00556, 2016.
8. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.
9. J. Wen et al., "Applications of Convolutional Neural Networks to Classification of Alzheimer's Disease: A Review and Reproducibility Analysis," Frontiers in Neuroinformatics, 2020.
10. S. J. Pan and Q. Yang, "A survey on transfer learning," IEEE TKDE, 2010.
