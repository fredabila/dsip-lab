# DSIP Lab: train a medical image AI model for your class project

This folder lets your group **train, test and demo an image model** using topics such as breast cancer detection or malaria parasite detection, **without writing any training code**.

You describe your project in one small settings file (a YAML *config*). Then you run a few commands:

```
 your dataset ──► preview ──► prepare ──► train ──► evaluate ──► predict / demo
 (folders of     (see the     (split +    (teach     (how good   (use it on
  images)         filters)     filter)     the AI)    is it?)     new images)
```

The **signal processing** part of your project lives in the preprocessing step. That is where you choose filters such as CLAHE, Fourier filters and denoising, and then measure how they change the model's accuracy.

> ⚠️ This is for learning only. A model trained here is **not** a medical device and must never be used to diagnose real patients.

---

## Contents

1. [Choose how to run it](#1-choose-how-to-run-it)
2. [Option A: Kaggle or Colab (easiest, free GPU)](#option-a-kaggle-or-colab-easiest-free-gpu)
3. [Option B: your own laptop](#option-b-your-own-laptop)
4. [Your first model, step by step (malaria example)](#4-your-first-model-step-by-step-malaria-example)
5. [Setting up YOUR project](#5-setting-up-your-project)
6. [The signal-processing part: choosing filters](#6-the-signal-processing-part-choosing-filters)
7. [Understanding your results](#7-understanding-your-results)
8. [Experiments for your report](#8-experiments-for-your-report)
9. [Working as a group](#9-working-as-a-group)
10. [Troubleshooting](#10-troubleshooting)
11. [Command cheat sheet](#11-command-cheat-sheet)
12. [Glossary](#12-glossary)

---

## 1. Choose how to run it

Training is much faster on a GPU (a graphics card). Most student laptops don't have a suitable one, and that's fine: Kaggle and Google Colab give you one for free.

| | Option A: Kaggle / Colab | Option B: your laptop |
|---|---|---|
| Install anything? | No, it runs in your browser | Yes, Python and packages |
| GPU | Free | Only if you have an NVIDIA card |
| Training time (malaria example) | Minutes | Much longer on CPU (use a smaller subset) |
| Best for | Training the real model | Editing configs, previews, small tests, demos |

**Recommended for most groups:** set up the project and preview on a laptop, then do the real training on Kaggle or Colab. You can also do everything on Kaggle.

---

## Option A: Kaggle or Colab (easiest, free GPU)

Use the ready-made notebook [`notebooks/train_colab_kaggle.ipynb`](notebooks/train_colab_kaggle.ipynb).

**On Kaggle** (best when your dataset is already on Kaggle):

1. Go to kaggle.com → **Create → New Notebook** → **File → Import Notebook**, and upload `train_colab_kaggle.ipynb`.
2. In the right panel, open **Settings**, then:
   - set **Accelerator → GPU T4 x2** (or P100),
   - switch **Internet → On** (this needs a phone-verified account).
3. Click **+ Add Input** and search for your dataset, e.g. *"cell images for detecting malaria"*.
4. In the second code cell, set `SOURCE` to the dataset's folder, e.g.
   `SOURCE = "/kaggle/input/cell-images-for-detecting-malaria/cell_images"`.
   You can find the exact path by expanding the dataset in the right-hand **Input** panel.
5. Set `CONFIG` in the first cell, e.g. `"configs/malaria_cells.yaml"`.
6. Click **Run All**.

**On Google Colab:**

1. Go to colab.research.google.com → **File → Upload notebook**, and upload `train_colab_kaggle.ipynb`.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Get the dataset into Colab, either:
   - with the Kaggle CLI: uncomment the lines in the dataset cell and upload your `kaggle.json` token (see [Getting datasets](#getting-datasets)), or
   - by uploading it to Google Drive and mounting Drive.
4. Run the cells from top to bottom.

At the end, the notebook zips your trained run (weights, plots and metrics) so you can download it.

> 💡 Free sessions disconnect after a few hours of inactivity, and their files are deleted. **Download your run zip before closing the tab.**

---

## Option B: your own laptop

### What you need

- **Python 3.10 or newer.** Check with `python --version`. If needed, download it from python.org; on Windows, tick **"Add Python to PATH"** during install.
- **Git**, to clone the repo, or download the repo as a ZIP from GitHub.
- **About 5–10 GB of free disk space** for datasets and runs.
- An internet connection the first time you train: pretrained model weights download automatically.

### Install (one time)

Open a terminal (on Windows: **PowerShell**), then run:

```bash
git clone https://github.com/fredabila/dsip-lab.git
cd dsip-lab/model
python -m venv .venv
```

Activate the virtual environment. **You must do this every time you open a new terminal.**

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Then install the toolkit:

```bash
pip install -e ".[dev,demo]"
```

This takes a few minutes, because it installs PyTorch. To check that it worked, run:

```bash
dsip --help
dsip ops
```

> **Have an NVIDIA GPU on Windows?** Before the `pip install` line above, install the CUDA version of PyTorch using the command from pytorch.org/get-started/locally. Otherwise you get the slower CPU version.
>
> **`dsip` not recognised?** Use `python -m dsip` instead. It does exactly the same thing (e.g. `python -m dsip ops`).

**Always run `dsip` commands from inside the `model/` folder.** All paths in the configs are relative to it.

---

## 4. Your first model, step by step (malaria example)

This example uses the NIH malaria dataset: 27,558 microscope images of single red blood cells, each labelled **Parasitized** or **Uninfected**.

### Step 0: Download the dataset

Download it from Kaggle: <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria> (the **Download** button). Unzip it so that you end up with:

```
model/
  data/
    raw/
      cell_images/
        Parasitized/   C33P1thinF_IMG_..._cell_170.png ...
        Uninfected/    C1_thinF_IMG_..._cell_128.png ...
```

That folder, `data/raw/cell_images`, is what `configs/malaria_cells.yaml` expects.

If your folder is somewhere else, either edit `source:` in the config or add `--source "path/to/cell_images"` to the `preview` and `prepare` commands.

> 💡 **First time? Do a quick test run.** Open `configs/malaria_cells.yaml` and change `max_per_class: null` to `max_per_class: 500` and `epochs: 25` to `epochs: 3`. You'll get a (bad) model in a few minutes and know everything works. Change them back for the real run.

### Step 1: Preview the filters

```bash
dsip preview -c configs/malaria_cells.yaml --spectrum
```

This saves `figures/malaria_cells_preprocess.png`. Each row is a sample image; the columns show the original, its Fourier spectrum, the image after each filter, and the final spectrum. **This is a ready-made figure for your report.**

### Step 2: Prepare the dataset

```bash
dsip prepare -c configs/malaria_cells.yaml
```

This step:

- finds every image and its label (taken from the folder name),
- removes duplicate files,
- splits the images into **train** (70%), **val** (15%) and **test** (15%),
- applies your filters to every image and saves the results in `data/prepared/malaria_cells/`.

At the end it prints how many images went into each split. Check that the numbers look sensible.

### Step 3: Train

```bash
dsip train -c configs/malaria_cells.yaml
```

You'll see one progress line per **epoch** (one full pass over the training images). Training stops early if the model stops improving. When it finishes, it **automatically tests the model** on the test set and prints the results.

Everything is saved in `runs/malaria_cells/<date-time>/`.

### Step 4: Look at the results

Open `runs/malaria_cells/<date-time>/eval_test/`. It contains:

- `metrics.json`: all the numbers,
- `confusion_matrix.png` and `roc_curve.png`: figures for your report,
- `predictions.csv`: one row per test image. Sort by the `correct` column to find the mistakes, then look at those images.

[Section 7](#7-understanding-your-results) explains what the numbers mean.

### Step 5: Use the model

```bash
# predict on any image or folder
dsip predict -c configs/malaria_cells.yaml path/to/some/images/

# live web demo for your presentation (opens in the browser at http://127.0.0.1:7860)
dsip demo -c configs/malaria_cells.yaml
```

With `-c config.yaml`, these commands automatically use the **most recent** run of that config. To pick a specific run, use `--run runs/malaria_cells/<date-time>`.

---

## 5. Setting up YOUR project

### Start from an existing config

| Your topic | Start from | Dataset |
|---|---|---|
| Malaria: is this cell infected? | `configs/malaria_cells.yaml` | [NIH cell images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |
| Malaria: find and count parasites in a whole smear | `configs/malaria_detection.yaml` | A YOLO-format dataset (e.g. a Roboflow Universe export) |
| Breast cancer, histopathology (tissue under a microscope) | `configs/breast_histopathology_idc.yaml` | [Breast histopathology images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) |
| Breast cancer, ultrasound | `configs/breast_ultrasound.yaml` | [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| Anything else (TB/pneumonia X-rays, skin lesions, retina, brain MRI, blood cells, ...) | `configs/_template.yaml` | Your choice |

Copy the closest one and give it your own name:

```bash
# Windows
copy configs\_template.yaml configs\my_project.yaml
# macOS / Linux
cp configs/_template.yaml configs/my_project.yaml
```

Then edit it in any text editor (VS Code, Notepad, ...).

> YAML tips: indentation matters (use **spaces, not tabs**), and `#` starts a comment.

### Arrange your images

**For classification** (one answer per image, e.g. benign or malignant), each image must be inside a folder named after its class:

```
data/raw/my_dataset/
    benign/
        img001.png
        img002.png
    malignant/
        img101.png
```

Deeper nesting is fine (e.g. `patient_12/malignant/img.png`): the **folder directly containing the image** is what counts.

**For detection** (boxes around each object, e.g. every parasite in a smear), use YOLO format. Each image has a `.txt` file with the same name in a `labels` folder:

```
data/raw/my_dataset/
    images/  smear01.png  smear02.png
    labels/  smear01.txt  smear02.txt
```

Each line of a `.txt` file is one object: `class_number center_x center_y width height`, with the coordinates as fractions of the image size (0 to 1). Tools like Roboflow or CVAT can export this format directly. Folders that are already split (`train/`, `valid/`, `test/`), as in a Roboflow export, are kept as they are.

### The config, explained

| Setting | What it does | Example |
|---|---|---|
| `name` | Your project's name. Output folders use it. **Use a new name for each experiment.** | `malaria_clahe` |
| `task` | `classify` (one label per image) or `detect` (boxes) | `classify` |
| `data.source` | Where your raw images are | `data/raw/cell_images` |
| `data.class_map` | Renames folders to class names. **Folders not listed are ignored.** | `"0": benign` |
| `data.exclude` | Skips files matching a pattern, e.g. segmentation masks | `["*_mask*"]` |
| `data.group_level` | Keeps all images of one patient in the same split (see below) | `1` |
| `data.split` | Fraction of images held out for validation and test | `{val: 0.15, test: 0.15}` |
| `data.balance` | Fixes unequal class sizes: `none`, `oversample` (copy the smaller class) or `undersample` (drop part of the bigger class) | `oversample` |
| `data.max_per_class` | Uses only N images per class: great for quick tests | `500` |
| `positive_class` | Which class means "disease". Turns on sensitivity and specificity. | `malignant` |
| `preprocess` | Your filters, applied in order (see [section 6](#6-the-signal-processing-part-choosing-filters)) | |
| `model` | Pretrained starting model: `yolo11n-cls.pt` is small and fast, `yolo11s-cls.pt` and `yolo11m-cls.pt` are bigger and usually better. For detection use `yolo11n.pt` etc. | `yolo11n-cls.pt` |
| `train.epochs` | Maximum number of passes over the data | `30` |
| `train.imgsz` | Size the images are resized to for the model (a multiple of 32) | `224` |
| `train.batch` | Images processed at once. Lower it if you run out of memory. | `32` |
| `train.patience` | Stop early after this many epochs without improvement | `10` |

Anything else under `train:` is passed straight to Ultralytics; see the full list at <https://docs.ultralytics.com/modes/train/#train-settings>.

### Patient-level splitting (very important for medical data)

Many datasets contain **several images from the same patient**. If one patient's images land in both train and test, the model can recognise the *patient* instead of the *disease*. Your test score then looks amazing but is fake.

If the dataset's folders identify the patient, set `group_level`:

```
data/raw/idc/10253/0/img.png        ← patient folder "10253", label folder "0"
              ▲     ▲
              │     └── label folder: group_level 0
              └──────── its parent:   group_level 1   ← use this
```

`breast_histopathology_idc.yaml` already does this.

---

## 6. The signal-processing part: choosing filters

Filters are listed under `preprocess:` and run **top to bottom**. The same filters are automatically applied to new images at prediction time.

```yaml
preprocess:
  - homomorphic: {gamma_low: 0.6, gamma_high: 1.4, cutoff: 0.05}
  - clahe: {clip_limit: 2.0, tile_grid: 8}
  - median_blur: {ksize: 3}
```

Run `dsip ops` to see every filter with all its parameters.

### What each filter does

| Filter | Domain | In plain words |
|---|---|---|
| `gamma` | Intensity | Brightens (gamma < 1) or darkens (gamma > 1) mid-tones |
| `contrast_stretch` | Intensity | Stretches the brightness range to use the full 0–255 |
| `hist_eq` | Histogram | Spreads out brightness values evenly over the whole image |
| `clahe` | Histogram | Like `hist_eq`, but works in small tiles and limits noise boosting. The go-to contrast enhancer for medical images. |
| `grayscale` | Colour | Removes colour |
| `channel` | Colour | Keeps one colour channel, e.g. `{name: green}`, which often shows stained parasites best |
| `reinhard_normalize` | Colour | Makes stain colours consistent across slides and labs (needs a reference image) |
| `invert` | Intensity | Negative image |
| `gaussian_blur` | Spatial | Smooths noise (a low-pass filter), but also blurs edges |
| `median_blur` | Spatial | Removes salt-and-pepper noise while keeping edges |
| `bilateral` | Spatial | Smooths flat areas, keeps edges sharp |
| `nl_means` | Spatial | Strong denoising, good for ultrasound speckle (slow) |
| `unsharp_mask` | Spatial | Sharpens by adding back detail |
| `laplacian_sharpen` | Spatial | Sharpens using the second derivative |
| `sobel` | Spatial | Turns the image into an edge map |
| `fft_filter` | Frequency | Butterworth filter in the Fourier domain: `lowpass`, `highpass`, `bandpass`, `bandstop` or `highboost` |
| `homomorphic` | Frequency | Fixes uneven lighting (e.g. dark corners in microscope images) |
| `tophat_enhance` | Morphology | Highlights small bright or dark spots, such as parasites or calcifications |
| `resize` | Geometry | Changes the image size |

### Good starting points by image type

| Image type | Try | Why |
|---|---|---|
| Blood smears (malaria) | `homomorphic`, `clahe`, `channel: green`, `tophat_enhance` | Uneven microscope lighting; small dark parasites |
| H&E histopathology | `reinhard_normalize`, `unsharp_mask` | Stain colour varies a lot between labs |
| Ultrasound | `median_blur` or `nl_means`, then `clahe` | Speckle noise; low contrast |
| X-ray / mammogram | `clahe`, `unsharp_mask`, `fft_filter: highboost` | Low contrast; fine details matter |

For **detection** projects, use only filters that don't move pixels around. Everything in the table is fine except `resize: {keep_aspect: true}`, which is also fine because it adds no padding. Your box labels stay correct.

**Want to write your own filter?** Add a function to [`dsip/preprocess.py`](dsip/preprocess.py):

```python
@register("my_filter")
def my_filter(img, strength: float = 1.0):
    """One line describing it (shows up in `dsip ops`)."""
    # img is a uint8 BGR numpy array (height x width x 3); return the same kind
    return img
```

Then use `- my_filter: {strength: 2.0}` in your config.

> After changing `preprocess:`, re-run `dsip prepare ... --overwrite`. The prepared images are saved on disk, so the old filters stay in place until you rebuild them.

---

## 7. Understanding your results

Example output from `dsip evaluate`:

```
Accuracy           0.9620
Balanced accuracy  0.9619
roc_auc            0.9910

class                precision    recall        f1       n
parasitized              0.970     0.953     0.961    2067
uninfected               0.954     0.971     0.962    2067

[parasitized @ threshold 0.500] sensitivity=0.953 specificity=0.971 PPV=0.970 NPV=0.954
[parasitized @ threshold 0.214] sensitivity=0.950 ...
```

*(Illustrative numbers. Yours will differ.)*

| Metric | Question it answers | Why it matters |
|---|---|---|
| **Sensitivity** (= recall of the disease class) | Of all sick patients, how many did we catch? | Missing a disease (a false negative) is usually the worst mistake |
| **Specificity** | Of all healthy patients, how many did we correctly clear? | Too low means many false alarms |
| **PPV** (precision) | When the model says "sick", how often is it right? | |
| **NPV** | When the model says "healthy", how often is it right? | |
| **Accuracy** | What fraction of predictions are correct? | **Misleading if classes are unequal**: with 95% healthy images, always answering "healthy" scores 95% |
| **Balanced accuracy** | The average of the recall of each class | Use this instead of accuracy when classes are unequal |
| **ROC-AUC** | How well the model ranks sick above healthy, across all thresholds | 0.5 is coin-flipping, 1.0 is perfect |

**The threshold:** the model outputs a probability, e.g. "0.7 chance this cell is parasitized". By default, 0.5 or above counts as positive.

- A **lower threshold** catches more disease (higher sensitivity) but raises more false alarms (lower specificity).
- `dsip evaluate` also shows the threshold that reaches 95% sensitivity. Try your own with `dsip evaluate -c ... --threshold 0.3`.

**Detection projects** report **mAP50** instead: roughly, how well the boxes match the real objects (0 to 1, higher is better). It also reports precision and recall.

**Figures to put in your report:**

- `figures/<name>_preprocess.png`: your filter pipeline,
- `eval_test/confusion_matrix.png`: which classes get confused,
- `eval_test/roc_curve.png`: the sensitivity/specificity trade-off,
- `results.png` (in the run folder): training curves. If the training loss keeps falling while validation gets worse, the model is **overfitting**.

> 🚩 **Results that look too good** (e.g. 99.9%) usually mean something leaked: duplicate images, or the same patient in train and test. Check the `group_level` setting.

---

## 8. Experiments for your report

Since this is a signal-processing class, the most interesting question is usually: **does the preprocessing help?** Keep everything else the same and change only the filters.

1. **Baseline.** Copy your config to `configs/exp_raw.yaml`, set `name: exp_raw`, and keep only `resize` in `preprocess`.
2. **Variants.** Make `exp_clahe.yaml`, `exp_homomorphic.yaml`, `exp_fft.yaml` and so on, each with a **different `name`** and different filters.
3. **Run each one:**

   ```bash
   dsip prepare -c configs/exp_raw.yaml
   dsip train   -c configs/exp_raw.yaml
   ```

4. **Compare them in one table:**

   ```bash
   dsip compare --out comparison.csv
   ```

   This prints the accuracy, balanced accuracy, AUC, sensitivity and specificity of every run side by side, and saves them as a CSV you can paste into your report.

Other ideas:

- Small vs. bigger model (`yolo11n-cls.pt` vs `yolo11s-cls.pt`).
- Image size (`imgsz: 128` vs `224`).
- Balancing (`none` vs `oversample`).
- Noise robustness: add `gaussian_blur` or noise to the *test* images only and see how far accuracy drops.

> Keep `seed` the same across experiments so the splits are identical and the comparison is fair.

---

## 9. Working as a group

- **One person sets up**, pushes the config to the group repo, and everyone else pulls it.
- **Don't commit datasets, runs or weights.** They are large and already ignored by git. Share trained runs as a zip (Google Drive, etc.) instead.
- **Give every experiment its own config file and `name`,** so results never overwrite each other and you can always see exactly what produced a number.
- A good split of work:
  - one person handles data (download, folder layout, `prepare`),
  - one handles the preprocessing experiments and figures,
  - one handles training on Kaggle/Colab,
  - one handles evaluation, error analysis (`predictions.csv`) and the demo.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `dsip: command not found` / not recognised | Activate the virtual environment (see [Option B](#option-b-your-own-laptop)), or use `python -m dsip ...` |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Skip activation and run the venv's Python directly: `.venv\Scripts\python -m dsip ...`. Or use Command Prompt with `.venv\Scripts\activate.bat` |
| `Dataset source not found` | The path in `data.source` or `--source` is wrong. Paths are relative to the `model/` folder. Quote paths with spaces. |
| `No labelled images found` | Your images aren't inside class-named folders, or `class_map` doesn't match the folder names exactly (it is case-sensitive) |
| `WARNING: classes with < 10 images (wrong folder?)` | `source` points one level too high or too low. Point it at the folder that directly contains the class folders. |
| `already exists. Use --overwrite` | You've prepared this config before. Add `--overwrite` to rebuild it. |
| `No training images for class(es)` | Too few images (or patients) for the split. Use more data or a smaller `split`. |
| `No trained runs found` | Run `dsip train` first, or pass `--run <folder>` |
| `CUDA out of memory` | Lower `train.batch` (e.g. 16 or 8) or `imgsz` |
| Training is very slow | You're on CPU. Use Kaggle/Colab, or set `max_per_class: 1000` and a smaller `imgsz` |
| Kaggle CLI says `401 Unauthorized` | Put your `kaggle.json` token in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`). Get it from Kaggle → Settings → **Create New Token**. |
| Download of `yolo11n-cls.pt` fails | The first training run needs internet (on Kaggle, switch **Internet** on) |
| Scores are suspiciously perfect | Data leakage: see [patient-level splitting](#patient-level-splitting-very-important-for-medical-data) |
| `demo` says gradio is missing | `pip install gradio` |
| `.dcm` (DICOM) files fail to load | `pip install pydicom` |

Something else? Run the self-test: `pip install -e ".[dev]"` then `pytest`. It trains a tiny model on fake images, takes about a minute, and tells you whether the installation works.

---

## Getting datasets

Many medical datasets are on Kaggle. You can download them in two ways:

- **In the browser:** open the dataset page → **Download** → unzip it into `model/data/raw/`.
- **With the command line:**
  1. `pip install kaggle`
  2. On Kaggle, go to your profile → **Settings** → **API → Create New Token**. This downloads `kaggle.json`.
  3. Move it to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
  4. Then download, for example:

     ```bash
     kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p data/raw --unzip
     ```

Always read the dataset's **license** and cite it in your report.

---

## 11. Command cheat sheet

All commands are run from the `model/` folder. `-c` means `--config`.

```bash
dsip ops                                        # list all filters and their parameters
dsip preview  -c configs/X.yaml --spectrum      # figure of the filter pipeline
dsip preview  -c configs/X.yaml --images a.png b.png   # ...on specific images
dsip prepare  -c configs/X.yaml [--overwrite] [--source PATH]
dsip train    -c configs/X.yaml [--epochs 10] [--batch 16] [--device 0|cpu] [--model yolo11s-cls.pt] [--no-eval]
dsip evaluate -c configs/X.yaml [--threshold 0.3] [--target-sensitivity 0.9] [--split val]
dsip predict  -c configs/X.yaml path/to/images_or_folder [--threshold 0.3]
dsip compare  [--out comparison.csv]            # metrics of all runs side by side
dsip export   -c configs/X.yaml --format onnx   # or tflite, torchscript, ...
dsip demo     -c configs/X.yaml [--share]       # --share gives a public link (useful on Colab)
```

Add `--run runs/<name>/<date-time>` to `evaluate`, `predict`, `export` or `demo` to use a specific run instead of the latest one.

### Where things end up

```
model/
  data/raw/<dataset>/                 your downloaded datasets (you put them here)
  data/prepared/<name>/               split + filtered images, manifest.csv
  figures/<name>_preprocess.png       filter preview
  runs/<name>/<date-time>/
      weights/best.pt                 ← your trained model
      dsip_config.yaml                exact settings used
      results.png                     training curves
      eval_test/                      metrics.json, confusion matrix, ROC, predictions.csv
      predictions/                    output of `dsip predict`
```

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Classification** | One label per image (e.g. infected / uninfected) |
| **Detection** | Boxes drawn around each object in the image, with a label each |
| **Train / val / test split** | The model learns from *train*, the best version is picked using *val*, and the final score comes from *test*, which the model has never seen |
| **Epoch** | One pass over all the training images |
| **Batch size** | How many images are processed together in one step |
| **Pretrained model / transfer learning** | We start from a model already trained on millions of everyday photos and fine-tune it on medical images, which works far better than starting from zero |
| **Overfitting** | The model memorises the training images and does badly on new ones |
| **Data leakage** | Test information sneaks into training (duplicates, the same patient), making scores look better than they really are |
| **Augmentation** | Random flips and colour changes during training, so the model sees more variety |
| **Threshold** | The probability above which we call an image "positive" |
| **YOLO11 / Ultralytics** | The model family and library doing the actual training underneath |
