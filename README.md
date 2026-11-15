# DSIP Lab

**Train medical image AI models for your Digital Signal & Image Processing class project, without writing training code.**

Pick a topic, such as breast cancer detection or malaria parasite detection, and point the toolkit at a folder of images. It then:

1. **cleans up the images** with classic signal-processing filters: contrast enhancement (CLAHE), Fourier-domain filters, denoising, stain normalisation, and more,
2. **trains** a modern image model (YOLO11) on them,
3. **tells you how good it is**, using the measures doctors use: sensitivity, specificity, ROC curves and confusion matrices,
4. **lets you use it** on new images, or show it off in a live web demo.

```
 folders of   ──►  preview  ──►  prepare  ──►  train  ──►  evaluate  ──►  predict / demo
   images         (filters)    (split+filter)              (metrics)
```

> ⚠️ For learning only. Models trained here are not medical devices and must never be used to diagnose real patients.

---

## Start here

👉 **The full step-by-step guide is in [model/README.md](model/README.md).** It covers installation, your first model, setting up your own project, choosing filters, reading the results, and troubleshooting.

### Fastest way to try it (no installation)

Open [`model/notebooks/train_colab_kaggle.ipynb`](model/notebooks/train_colab_kaggle.ipynb) in **Kaggle** or **Google Colab**. Both give you a free GPU. Turn the GPU on, choose a project config, and click **Run All**. Instructions are inside the notebook.

### On your own computer

You need Python 3.10 or newer. On Windows, run these commands in PowerShell:

```bash
git clone https://github.com/fredabila/dsip-lab.git
cd dsip-lab/model
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,demo]"
dsip --help
```

Then download a dataset and follow [section 4 of the guide](model/README.md#4-your-first-model-step-by-step-malaria-example).

---

## Ready-made projects

| Topic | What the model does | Config |
|---|---|---|
| **Malaria** | Says whether a single blood cell is infected or not | `model/configs/malaria_cells.yaml` |
| **Malaria** | Finds and counts parasites in a whole blood-smear image | `model/configs/malaria_detection.yaml` |
| **Breast cancer** | Spots invasive cancer (IDC) in tissue images under a microscope | `model/configs/breast_histopathology_idc.yaml` |
| **Breast cancer** | Classifies ultrasound scans as benign, malignant or normal | `model/configs/breast_ultrasound.yaml` |
| **Your own topic** | Anything with folders of labelled images: X-rays, skin lesions, retina scans, MRI, ... | `model/configs/_template.yaml` |

Each config says which public dataset to download and where to put it.

---

## What's in this repository

```
dsip-lab/
├── model/                      ← the DSIP Lab toolkit (this is what you use)
│   ├── README.md               full student guide
│   ├── configs/                one settings file per project
│   ├── dsip/                   the toolkit's code (filters in dsip/preprocess.py)
│   ├── notebooks/              Kaggle/Colab notebook
│   ├── tests/                  self-test (run: pytest)
│   └── train/                  older example: PPE (hard-hat) detection notebook
├── backend/                    example: serving a trained model as a web API
└── mobile/                     example: phone app that calls that API
```

You only need the **`model/`** folder for a class project. `backend/` and `mobile/` are an optional bonus, described below.

---

## Bonus: deploying a model to a phone (example app)

`backend/` and `mobile/` contain a complete, working example of putting a trained YOLO model into a real product. It is a safety-gear (PPE) detector: a phone camera sends frames to a web server, which draws boxes around hard hats and safety vests.

It is not needed for the DSIP projects. Groups that want to show their model running live on a phone can use it as a reference: swap in your own `best.pt` and class names.

<details>
<summary>How to run the example app</summary>

**Backend** (Python web API, FastAPI + Ultralytics; uses the [uv](https://docs.astral.sh/uv/) package manager):

```bash
cd backend
uv sync
uv run fastapi dev main.py
```

It serves `POST /api/detect` (send an image, get detections back) and `GET /health`. The model file is `backend/models/helmax_ppe.pt`; that name is left over from the project this example came from. A `Dockerfile` is included for deployment.

**Mobile** (Expo / React Native):

```bash
cd mobile
npm install
npx expo start
```

Before starting, create a file `mobile/.env.local` with one line giving your backend's address, e.g. `EXPO_PUBLIC_API_URL=http://192.168.1.20:8000`.

| Variable | Description |
|---|---|
| `EXPO_PUBLIC_API_URL` | Address of the backend, e.g. `http://192.168.1.20:8000` |

Detected classes: hard hat and safety vest (compliant); no hard hat and no safety vest (violation).

</details>

---

## Credits and licence notes

- Training is built on [Ultralytics YOLO](https://docs.ultralytics.com/) (AGPL-3.0).
- Datasets belong to their authors. Check each dataset's licence and cite it in your report.
