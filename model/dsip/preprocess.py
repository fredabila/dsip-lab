"""Classic digital signal / image processing operations used as a preprocessing pipeline.

Every operation takes an 8-bit BGR image (H x W x 3, uint8) and returns one, so
operations can be chained in any order from a YAML config:

    preprocess:
      - clahe: {clip_limit: 2.0, tile_grid: 8}
      - median_blur: {ksize: 3}
      - resize: {size: 224}

Run ``dsip ops`` to list every operation and its parameters.

To add your own operation, write a function ``def my_op(img, **params) -> img``
and decorate it with ``@register("my_op")``. That's all.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OPS: dict[str, Callable[..., np.ndarray]] = {}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm"}


def register(name: str):
    def wrap(fn):
        OPS[name] = fn
        return fn

    return wrap


# --------------------------------------------------------------------------- I/O


def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce any image (gray, 16-bit, float, BGRA) to 8-bit, 3-channel BGR."""
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as 8-bit BGR. Handles non-ASCII paths, 16-bit files and DICOM."""
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError as e:  # pragma: no cover
            raise ImportError("Reading DICOM needs pydicom: pip install pydicom") from e
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr  # invert so that bright = dense tissue
        return to_bgr_uint8(arr)

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return to_bgr_uint8(img)


def save_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))


# ------------------------------------------------------------------- helpers


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _per_channel(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.dstack([fn(img[..., c]) for c in range(img.shape[2])])


def _rescale_uint8(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)


def _radial_distance(shape: tuple[int, int]) -> np.ndarray:
    """Distance of each frequency bin from the (centred) DC component, normalised to [0, 1]."""
    h, w = shape
    u = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    v = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    d = np.sqrt(u**2 + v**2)
    return d / d.max()


def log_spectrum(img: np.ndarray) -> np.ndarray:
    """Log-magnitude Fourier spectrum of the image's luminance (for visualisation)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    return _rescale_uint8(np.log1p(np.abs(f)))


# ---------------------------------------------------------- geometry / colour


@register("resize")
def resize(img: np.ndarray, size: int = 224, keep_aspect: bool = False) -> np.ndarray:
    """Resize to size x size, or scale the longest side to `size` when keep_aspect is true.

    No padding is added, so YOLO (normalised) box labels stay valid.
    """
    h, w = img.shape[:2]
    if keep_aspect:
        s = size / max(h, w)
        new = (max(1, round(w * s)), max(1, round(h * s)))
    else:
        new = (size, size)
    interp = cv2.INTER_AREA if new[0] < w else cv2.INTER_CUBIC
    return cv2.resize(img, new, interpolation=interp)


@register("grayscale")
def grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale (kept as 3 identical channels so models still accept it)."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


@register("channel")
def channel(img: np.ndarray, name: str = "green") -> np.ndarray:
    """Keep a single colour channel (blue/green/red, or L/a/b of CIELAB, or H/S/V).

    The green channel often has the best contrast for stained blood smears.
    """
    name = name.lower()
    bgr = {"blue": 0, "green": 1, "red": 2}
    if name in bgr:
        c = img[..., bgr[name]]
    elif name in {"l", "a", "b"}:
        c = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., "lab".index(name)]
    elif name in {"h", "s", "v"}:
        c = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., "hsv".index(name)]
    else:
        raise ValueError(f"Unknown channel '{name}'")
    return cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)


@register("invert")
def invert(img: np.ndarray) -> np.ndarray:
    """Photographic negative (255 - pixel)."""
    return 255 - img


# --------------------------------------------------------- intensity / contrast


@register("gamma")
def gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Power-law (gamma) transform: out = in^gamma. gamma < 1 brightens, > 1 darkens."""
    lut = (np.linspace(0, 1, 256) ** gamma * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


@register("contrast_stretch")
def contrast_stretch(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Linear contrast stretch between the given intensity percentiles (per channel)."""

    def f(c):
        lo, hi = np.percentile(c, [low_pct, high_pct])
        return ((c.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)

    return _per_channel(img, f)


@register("hist_eq")
def hist_eq(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation of the luminance channel (colour preserved)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


@register("clahe")
def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the L channel of CIELAB.

    Standard for mammograms, X-rays and low-contrast microscopy.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@register("reinhard_normalize")
def reinhard_normalize(
    img: np.ndarray,
    reference: str | None = None,
    target_mean: list[float] | None = None,
    target_std: list[float] | None = None,
) -> np.ndarray:
    """Reinhard colour (stain) normalisation: match LAB mean/std to a reference image.

    Reduces colour variation between labs/scanners in H&E histopathology and
    Giemsa-stained smears. Give either `reference` (path to a well-stained image)
    or explicit `target_mean`/`target_std` (3 values each, OpenCV 8-bit LAB scale).
    """
    if reference is not None:
        t_mean, t_std = _lab_stats(_reference_image(reference))
    elif target_mean is not None and target_std is not None:
        t_mean, t_std = np.array(target_mean, np.float32), np.array(target_std, np.float32)
    else:
        raise ValueError("reinhard_normalize needs `reference` or `target_mean` + `target_std`")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean, std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
    lab = (lab - mean) / (std + 1e-6) * t_std + t_mean
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


_REF_CACHE: dict[str, np.ndarray] = {}


def _reference_image(path: str) -> np.ndarray:
    if path not in _REF_CACHE:
        _REF_CACHE[path] = load_image(path)
    return _REF_CACHE[path]


def _lab_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    return lab.mean(0), lab.std(0)


# ----------------------------------------------------- spatial-domain filtering


@register("gaussian_blur")
def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Gaussian low-pass filter (smooths noise, also smooths edges)."""
    k = _odd(ksize)
    return cv2.GaussianBlur(img, (k, k), sigma)


@register("median_blur")
def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter: removes salt-and-pepper noise while keeping edges."""
    return cv2.medianBlur(img, _odd(ksize))


@register("bilateral")
def bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Edge-preserving bilateral filter."""
    return cv2.bilateralFilter(img, int(d), float(sigma_color), float(sigma_space))


@register("nl_means")
def nl_means(img: np.ndarray, h: float = 10, template: int = 7, search: int = 21) -> np.ndarray:
    """Non-local means denoising (slow but strong; good for ultrasound speckle)."""
    return cv2.fastNlMeansDenoisingColored(img, None, float(h), float(h), int(template), int(search))


@register("unsharp_mask")
def unsharp_mask(img: np.ndarray, amount: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    """Sharpen: out = img + amount * (img - gaussian(img))."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


@register("sobel")
def sobel(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gradient magnitude (Sobel) edge map."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=_odd(ksize))
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=_odd(ksize))
    return cv2.cvtColor(_rescale_uint8(np.hypot(gx, gy)), cv2.COLOR_GRAY2BGR)


@register("laplacian_sharpen")
def laplacian_sharpen(img: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Sharpen by subtracting the Laplacian (second derivative)."""
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3)
    return (img.astype(np.float32) - weight * lap).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------- morphology


@register("tophat_enhance")
def tophat_enhance(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Morphological contrast enhancement: img + white_tophat - black_tophat.

    Boosts small bright and dark structures (e.g. parasites, micro-calcifications)
    that are smaller than the structuring element.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(ksize), _odd(ksize)))
    white = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
    black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)
    return cv2.subtract(cv2.add(img, white), black)


# ------------------------------------------------------- frequency domain


@register("fft_filter")
def fft_filter(
    img: np.ndarray,
    kind: str = "lowpass",
    cutoff: float = 0.1,
    cutoff_high: float = 0.5,
    order: int = 2,
    boost: float = 1.0,
) -> np.ndarray:
    """Butterworth filter in the Fourier domain.

    kind: lowpass | highpass | bandpass | bandstop | highboost
    cutoff (and cutoff_high for band filters) are fractions (0..1) of the maximum frequency.
    highboost returns img + boost * highpass(img) (frequency-domain sharpening).
    """
    d = _radial_distance(img.shape[:2]) + 1e-8
    n = 2 * int(order)
    lp = 1.0 / (1.0 + (d / cutoff) ** n)
    hp = 1.0 - lp
    if kind == "lowpass":
        h = lp
    elif kind in {"highpass", "highboost"}:
        h = hp
    elif kind in {"bandpass", "bandstop"}:
        lp_high = 1.0 / (1.0 + (d / cutoff_high) ** n)
        h = lp_high * hp
        if kind == "bandstop":
            h = 1.0 - h
    else:
        raise ValueError(f"Unknown fft_filter kind '{kind}'")

    def f(c):
        spec = np.fft.fftshift(np.fft.fft2(c.astype(np.float32)))
        out = np.real(np.fft.ifft2(np.fft.ifftshift(spec * h)))
        if kind == "highboost":
            return (c.astype(np.float32) + boost * out).clip(0, 255).astype(np.uint8)
        if kind in {"highpass", "bandpass"}:
            return _rescale_uint8(out)
        return out.clip(0, 255).astype(np.uint8)

    return _per_channel(img, f)


@register("homomorphic")
def homomorphic(
    img: np.ndarray, gamma_low: float = 0.5, gamma_high: float = 1.5, cutoff: float = 0.05, c: float = 1.0
) -> np.ndarray:
    """Homomorphic filtering: corrects uneven illumination (vignetting in microscopy).

    Works on log(intensity): attenuates low frequencies (illumination) by gamma_low,
    amplifies high frequencies (reflectance/detail) by gamma_high.
    """
    d = _radial_distance(img.shape[:2])
    h = (gamma_high - gamma_low) * (1 - np.exp(-c * (d**2) / (cutoff**2))) + gamma_low
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = np.log1p(ycc[..., 0].astype(np.float32))
    spec = np.fft.fftshift(np.fft.fft2(y))
    out = np.expm1(np.real(np.fft.ifft2(np.fft.ifftshift(spec * h))))
    ycc[..., 0] = _rescale_uint8(out)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------- pipeline


def _parse_step(step: Any) -> tuple[str, dict]:
    if isinstance(step, str):
        return step, {}
    if isinstance(step, dict) and len(step) == 1:
        name, params = next(iter(step.items()))
        return name, dict(params or {})
    raise ValueError(f"Bad preprocess step {step!r}: use 'op_name' or {{op_name: {{param: value}}}}")


class Pipeline:
    """An ordered list of preprocessing operations built from the YAML config."""

    def __init__(self, steps: list[Any] | None = None):
        self.steps: list[tuple[str, dict]] = []
        for step in steps or []:
            name, params = _parse_step(step)
            if name not in OPS:
                raise ValueError(f"Unknown preprocessing op '{name}'. Available: {', '.join(sorted(OPS))}")
            sig = inspect.signature(OPS[name])
            unknown = set(params) - set(list(sig.parameters)[1:])
            if unknown:
                raise ValueError(f"Op '{name}' got unknown parameter(s) {sorted(unknown)}")
            self.steps.append((name, params))

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for _, out in self.trace(img):
            img = out
        return img

    def trace(self, img: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (label, image) after each step: useful for figures in your report."""
        for name, params in self.steps:
            img = to_bgr_uint8(OPS[name](img, **params))
            args = ", ".join(f"{k}={v}" for k, v in params.items())
            yield (f"{name}({args})" if args else name), img

    def __repr__(self) -> str:
        return " -> ".join(n for n, _ in self.steps) or "(no preprocessing)"


def describe_ops() -> str:
    lines = []
    for name in sorted(OPS):
        fn = OPS[name]
        params = list(inspect.signature(fn).parameters.values())[1:]
        sig = ", ".join(f"{p.name}={p.default!r}" for p in params)
        doc = (inspect.getdoc(fn) or "").strip()
        lines.append(f"{name}({sig})\n    " + doc.replace("\n", "\n    "))
    return "\n\n".join(lines)
