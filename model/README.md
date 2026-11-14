# DSIP Lab: medical image model training

A small, config-driven toolkit for the Digital Signal & Image Processing class project.
You describe your project in one YAML file. The toolkit then:

1. **previews** your signal/image-processing pipeline (with Fourier spectra) for your report,
2. **prepares** the dataset: patient-level train/val/test split, de-duplication, class balancing, and preprocessing,
3. **trains** a pretrained YOLO11 classifier or detector (Ultralytics),
4. **evaluates** it with clinical metrics: sensitivity, specificity, PPV, NPV, ROC-AUC, and a confusion matrix,
5. **predicts** on new images, exports the model, or runs a **live web demo**.

> Educational use only. These models are not medical devices and must not be used for diagnosis.

## Quick start

```bash
cd model
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,demo]"
```

To use an NVIDIA GPU on Windows, first install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>. No GPU? Use the Colab/Kaggle notebook in [`notebooks/train_colab_kaggle.ipynb`](notebooks/train_colab_kaggle.ipynb).

The example below uses the malaria cell dataset:

```bash
# 1. download (needs a Kaggle account + API token, or download the zip by hand)
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p data/raw --unzip

# 2. look at what the preprocessing does (figure saved to figures/)
dsip preview -c configs/malaria_cells.yaml --spectrum

# 3. split + preprocess
dsip prepare -c configs/malaria_cells.yaml

# 4. train (also evaluates on the test set when finished)
dsip train -c configs/malaria_cells.yaml

# 5. later: evaluate / predict / demo using the latest run of that config
dsip evaluate -c configs/malaria_cells.yaml
dsip predict  -c configs/malaria_cells.yaml path/to/new_images/
dsip demo     -c configs/malaria_cells.yaml
```

`dsip` and `python -m dsip` are interchangeable. Run every command from the `model/` folder.

## Ready-made project configs

| Config | Topic | Task | Dataset |
|---|---|---|---|
| `configs/malaria_cells.yaml` | Malaria | Classify cell: parasitized / uninfected | [NIH malaria cell images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (27.5k) |
| `configs/malaria_detection.yaml` | Malaria | Detect parasites in whole smears | Any YOLO-format export, e.g. from Roboflow Universe |
| `configs/breast_histopathology_idc.yaml` | Breast cancer | Classify H&E patch: IDC / non-IDC | [Breast histopathology images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (277k, 279 patients) |
| `configs/breast_ultrasound.yaml` | Breast cancer | Classify ultrasound: benign / malignant / normal | [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) (780) |
| `configs/_template.yaml` | Your own | Either | Copy and edit: every option is documented inside |

Other topics fit the same pattern if the data is "folders of images named by class" or YOLO boxes. Examples are TB or pneumonia chest X-rays, skin lesions, diabetic retinopathy, brain-tumour MRI slices, and blood-cell types. For mammograms in DICOM, run `pip install pydicom`; `.dcm` files are then read automatically.

## Dataset layouts

**Classification**: the label is the name of the folder each image sits in, at any depth:

```
data/raw/my_dataset/
    benign/      img001.png ...
    malignant/   img101.png ...
```

Use `class_map` to rename folders (for example `"0": benign`) or to leave a class out.
If several images come from the same patient, set `group_level` so that a patient's images never land in both train and test (see the IDC config).

**Detection**: YOLO format. Each `images/*.png` has a matching `labels/*.txt` with one line per object: `class cx cy w h` (normalised 0-1). The data can be pre-split into `train/ valid/ test/` (as in a Roboflow "YOLOv8" export) or be a single pool that we split.

## The signal-processing part

Preprocessing is a list of operations in your config, applied in order. The same steps run at training and inference time, and the run folder saves them, so predictions always match training.

```yaml
preprocess:
  - homomorphic: {gamma_low: 0.6, gamma_high: 1.4, cutoff: 0.05}
  - clahe: {clip_limit: 2.0, tile_grid: 8}
  - median_blur: {ksize: 3}
```

`dsip ops` prints the full list with parameters:

| Domain | Operations |
|---|---|
| Intensity / histogram | `gamma`, `contrast_stretch`, `hist_eq`, `clahe`, `invert` |
| Colour | `grayscale`, `channel` (e.g. green channel for Giemsa smears), `reinhard_normalize` (stain normalisation) |
| Spatial filtering | `gaussian_blur`, `median_blur`, `bilateral`, `nl_means`, `unsharp_mask`, `laplacian_sharpen`, `sobel` |
| Frequency domain | `fft_filter` (Butterworth low/high/band-pass/band-stop/high-boost), `homomorphic` (illumination correction) |
| Morphology | `tophat_enhance` (boost small bright/dark structures) |
| Geometry | `resize` |

To add your own filter, write a function in [`dsip/preprocess.py`](dsip/preprocess.py) and decorate it with `@register("my_filter")`. It can then be used in any config.

### A suggested experiment for your report

Train the same model with different preprocessing and compare the test metrics. Use a different `name` per variant so the outputs don't overwrite each other:

```bash
cp configs/malaria_cells.yaml configs/malaria_raw.yaml           # edit: name: malaria_raw, preprocess: [resize only]
cp configs/malaria_cells.yaml configs/malaria_homomorphic.yaml   # edit: name: ..., add homomorphic
dsip prepare -c configs/malaria_raw.yaml && dsip train -c configs/malaria_raw.yaml
dsip prepare -c configs/malaria_homomorphic.yaml && dsip train -c configs/malaria_homomorphic.yaml
```

Then compare `runs/<name>/<run>/eval_test/metrics.json` across variants.

## Outputs

```
data/prepared/<name>/          split + preprocessed images, manifest.csv, prepare_summary.json
runs/<name>/<run>/
    weights/best.pt            trained model
    dsip_config.yaml           exact config used (preprocessing, classes)
    results.csv, *.png         Ultralytics training curves
    eval_test/
        metrics.json           accuracy, sensitivity, specificity, AUC, ...
        confusion_matrix*.png, roc_curve.png, pr_curve.png
        predictions.csv        per-image probabilities: find the misclassified cases
figures/<name>_preprocess.png  pipeline preview
```

Datasets, runs and weights are git-ignored, so don't commit them.

## Reading the metrics

- **Sensitivity (recall)**: the share of sick patients the model catches. Missing disease is usually the costliest error.
- **Specificity**: the share of healthy patients correctly cleared.
- **PPV / NPV**: when the model says positive or negative, how often it is right. Both depend on how common the disease is.
- **ROC-AUC**: threshold-independent ranking quality (0.5 is random, 1.0 is perfect).
- `dsip evaluate` also reports the threshold that reaches 95% sensitivity, and what specificity you pay for it. Change the target with `--target-sensitivity 0.9`, and try other thresholds with `--threshold 0.3`.
- Use **balanced accuracy**, not plain accuracy, on imbalanced data.

## Tips

- **Check the pipeline first.** Set `max_per_class: 500` and `epochs: 3` for a quick end-to-end run.
- **Bigger models** (`yolo11s-cls.pt`, `yolo11m-cls.pt`) are usually more accurate but slower.
- **Re-prepare after changing preprocessing:** `dsip prepare -c ... --overwrite`. Prepared images are cached on disk.
- **Out of GPU memory?** Lower `batch`.
- **Tests:** `pytest` runs a tiny synthetic end-to-end training on CPU (about 1 minute).
