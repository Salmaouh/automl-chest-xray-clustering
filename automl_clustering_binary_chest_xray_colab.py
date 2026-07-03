# ============================================================
# INTERNAL-ONLY AUTOML FOR BINARY CHEST X-RAY DATASET
# UPDATED VERSION
#
# Key updates:
#   1) Correct Davies–Bouldin reverse min–max normalization:
#        DB_norm = (DB_max - DB) / (DB_max - DB_min + eps)
#   2) Fixed warm-up normalization bounds used throughout BO
#   3) No clipping of normalized values
#   4) Normalization diagnostics added:
#        - checks whether BO trials exceed warm-up bounds
#        - counts normalized values outside [0,1]
#   5) Robust Drive-to-local dataset copy + safe local image loading
#      to avoid Google Drive DataLoader read failures
#   6) Added manuscript-ready visualizations:
#        - 2D UMAP projection of binary test embeddings by cluster
#        - Cluster × class count matrix for the binary test set
# ============================================================


# ============================================================
# 0) INSTALL REQUIRED PACKAGES
# ============================================================
!pip -q install timm umap-learn hdbscan bayesian-optimization


# ============================================================
# 1) IMPORTS
# ============================================================
import os
import math
import time
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile, UnidentifiedImageError
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import timm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
    OPTICS
)
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin

import umap
from bayes_opt import BayesianOptimization
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster as scipy_fcluster

warnings.filterwarnings("ignore")

try:
    import hdbscan
except ImportError:
    hdbscan = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 2) GLOBAL SETTINGS
# ============================================================
RNG = 42

random.seed(RNG)
np.random.seed(RNG)
torch.manual_seed(RNG)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)


# ============================================================
# 3) MOUNT GOOGLE DRIVE + COPY DATASET TO LOCAL COLAB STORAGE
# ============================================================
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/Chest_XRay_updated"
LOCAL_ROOT = "/content/Chest_XRay_updated_local"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
ID_TO_CLASS = {0: "Normal", 1: "Pneumonia"}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_dataset_root(base_dir):
    """
    Locate the dataset folder that directly contains train/val/test.
    """
    for root, dirs, files in os.walk(base_dir):
        dirs_lower = [d.lower() for d in dirs]
        if "train" in dirs_lower and "val" in dirs_lower and "test" in dirs_lower:
            return root
    return None


def count_images_in_dataset(root_dir):
    """
    Count images per split and class, returning a nested dictionary.
    """
    counts = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, split)
        split_total = 0
        split_counts = {}

        if not os.path.isdir(split_dir):
            return None

        for cname in CLASS_NAMES:
            class_dir = os.path.join(split_dir, cname)

            if not os.path.isdir(class_dir):
                return None

            n_imgs = sum(
                1 for fn in os.listdir(class_dir)
                if fn.lower().endswith(IMG_EXTS)
            )

            split_counts[cname] = n_imgs
            split_total += n_imgs

        counts[split] = {
            "total": split_total,
            "class_counts": split_counts
        }

    return counts


def copy_file_with_retries(src, dst, max_retries=5, delay=1.0):
    """
    Robustly copy one file from Drive to local storage.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            shutil.copy2(src, dst)
            return
        except Exception as e:
            last_error = e
            print(
                f"Retry {attempt}/{max_retries} while copying:\n"
                f"{src}\n"
                f"Reason: {repr(e)}"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to copy file after {max_retries} attempts:\n"
        f"{src}\n"
        f"Last error: {repr(last_error)}"
    )


def copy_dataset_to_local(src_root, dst_root):
    """
    Copy the full pre-split binary dataset to local Colab storage.
    """
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)

    print("\nCopying dataset from Drive to local Colab storage...")
    print("Source:", src_root)
    print("Destination:", dst_root)

    copied_files = 0

    for split in ["train", "val", "test"]:
        for cname in CLASS_NAMES:
            src_class_dir = os.path.join(src_root, split, cname)
            dst_class_dir = os.path.join(dst_root, split, cname)

            if not os.path.isdir(src_class_dir):
                raise FileNotFoundError(f"Missing folder: {src_class_dir}")

            os.makedirs(dst_class_dir, exist_ok=True)

            image_files = [
                fn for fn in sorted(os.listdir(src_class_dir))
                if fn.lower().endswith(IMG_EXTS)
            ]

            for fn in image_files:
                src_file = os.path.join(src_class_dir, fn)
                dst_file = os.path.join(dst_class_dir, fn)

                copy_file_with_retries(src_file, dst_file)
                copied_files += 1

            print(f"Copied {split}/{cname}: {len(image_files)} images")

    print(f"\nFinished copying {copied_files} images locally.")


print("\nChecking Drive dataset folder...")

if not os.path.exists(DRIVE_ROOT):
    raise FileNotFoundError(f"Dataset folder not found in Drive: {DRIVE_ROOT}")

detected_drive_root = find_dataset_root(DRIVE_ROOT)

if detected_drive_root is None:
    raise FileNotFoundError(
        "Could not find train/val/test folders inside the Drive dataset folder."
    )

DRIVE_ROOT = detected_drive_root

print("Detected Drive dataset root:", DRIVE_ROOT)
print("Drive dataset root contents:", os.listdir(DRIVE_ROOT))

drive_counts = count_images_in_dataset(DRIVE_ROOT)

if drive_counts is None:
    raise RuntimeError("Drive dataset structure is incomplete.")

print("\nDrive dataset counts:")
for split, info in drive_counts.items():
    print(split, "=", info["total"], "|", info["class_counts"])

local_counts = count_images_in_dataset(LOCAL_ROOT)

if local_counts != drive_counts:
    copy_dataset_to_local(DRIVE_ROOT, LOCAL_ROOT)
else:
    print("\nA complete local dataset copy already exists. Reusing it.")

local_counts = count_images_in_dataset(LOCAL_ROOT)

if local_counts != drive_counts:
    raise RuntimeError(
        "Local dataset copy does not match the Drive dataset counts. "
        "Please rerun this cell."
    )

print("\nLocal dataset counts verified:")
for split, info in local_counts.items():
    print(split, "=", info["total"], "|", info["class_counts"])

ROOT = LOCAL_ROOT
print("\nUsing local dataset root:", ROOT)


# ============================================================
# 4) INDEX PRE-SPLIT DATASET
# ============================================================
def collect_split(split_path):
    """
    Read file paths and numeric labels from a split folder.
    """
    paths, labels = [], []

    for cname in CLASS_NAMES:
        cdir = os.path.join(split_path, cname)
        assert os.path.isdir(cdir), f"Missing class folder: {cdir}"

        cid = CLASS_TO_ID[cname]

        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(cdir, fn))
                labels.append(cid)

    return np.array(paths), np.array(labels)


X_train, y_train = collect_split(os.path.join(ROOT, "train"))
X_val, y_val = collect_split(os.path.join(ROOT, "val"))
X_test, y_test = collect_split(os.path.join(ROOT, "test"))

print(f"\ntrain={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
print("Train:", Counter(y_train))
print("Val:", Counter(y_val))
print("Test:", Counter(y_test))


# ============================================================
# 5) DINO FEATURE EXTRACTION
# ============================================================
def safe_open_grayscale_image(path, max_retries=3, delay=0.25):
    """
    Safely open a local image file with retries.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with Image.open(path) as raw_img:
                raw_img.load()
                img = raw_img.convert("L")
            return img

        except (
            OSError,
            ConnectionAbortedError,
            ConnectionResetError,
            TimeoutError,
            UnidentifiedImageError
        ) as e:
            last_error = e
            print(
                f"Image read retry {attempt}/{max_retries} for:\n"
                f"{path}\n"
                f"Reason: {repr(e)}"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Could not read image after {max_retries} attempts:\n{path}\n"
        f"Last error: {repr(last_error)}"
    )


class ImgPathDS(Dataset):
    def __init__(self, paths, size=224):
        self.paths = list(paths)
        self.tf = tv.transforms.Compose([
            tv.transforms.Resize((size, size)),
            tv.transforms.Grayscale(num_output_channels=3),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = safe_open_grayscale_image(self.paths[i])
        return self.tf(img)


def build_dino_model(device):
    """
    Create frozen DINO ViT-S/16 feature extractor.
    """
    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    model.reset_classifier(0)
    model.to(device).eval()
    return model


DINO_MODEL = build_dino_model(DEVICE)


@torch.no_grad()
def dino_features(paths, batch_size=64, device=DEVICE, model=DINO_MODEL):
    """
    Extract DINO CLS-token features.
    """
    ds = ImgPathDS(paths)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    feats = []

    for xb in dl:
        xb = xb.to(device, non_blocking=(device == "cuda"))
        out = model.forward_features(xb)

        if isinstance(out, dict):
            fb = out.get(
                "x_norm_clstoken",
                out.get("cls_token", out.get("pooled", next(iter(out.values()))))
            )
        else:
            fb = out

        if fb.ndim == 3:
            fb = fb[:, 0, :]

        feats.append(fb.detach().cpu().numpy())

    return np.concatenate(feats, axis=0)


print("\nExtracting DINO features safely from local Colab storage...")
F_tr = dino_features(X_train)
F_val = dino_features(X_val)
F_te = dino_features(X_test)

print("DINO shapes:", F_tr.shape, F_val.shape, F_te.shape)


# ============================================================
# 6) STANDARDIZE + L2 NORMALIZE
# ============================================================
scaler = StandardScaler().fit(F_tr)


def zscore_l2(a):
    """
    Training-set z-score standardization followed by L2 normalization.
    """
    z = scaler.transform(a)
    n = np.sqrt(np.sum(z ** 2, axis=1, keepdims=True) + 1e-8)
    return z / n


Xtr = zscore_l2(F_tr)
Xval = zscore_l2(F_val)
Xte = zscore_l2(F_te)

print("\nNormalized feature shapes:")
print("Xtr:", Xtr.shape)
print("Xval:", Xval.shape)
print("Xte:", Xte.shape)


# ============================================================
# 7) UMAP DIMENSIONALITY REDUCTION
# ============================================================
umap_model = umap.UMAP(
    n_components=16,
    metric="cosine",
    n_neighbors=30,
    min_dist=0.05,
    random_state=RNG
).fit(Xtr)

Ztr = umap_model.transform(Xtr)
Zval = umap_model.transform(Xval)
Zte = umap_model.transform(Xte)

print("\nUMAP shapes:")
print("Ztr:", Ztr.shape)
print("Zval:", Zval.shape)
print("Zte:", Zte.shape)


# ============================================================
# 8) DEVELOPMENT POOL + FIXED 800-SAMPLE SEARCH SUBSET
# ============================================================
Z_full_train = np.vstack([Ztr, Zval])
y_full_train = np.concatenate([y_train, y_val])

subset_size = min(800, Z_full_train.shape[0])

rs = np.random.RandomState(RNG)
idx = rs.choice(
    Z_full_train.shape[0],
    subset_size,
    replace=False
)

Z_sub = Z_full_train[idx]

print("\nDevelopment/search shapes:")
print("Z_full_train:", Z_full_train.shape)
print("Z_sub:", Z_sub.shape)
print("Zte:", Zte.shape)


# ============================================================
# 9) INTERNAL METRICS + CORRECT NORMALIZATION
# ============================================================
def safe_internal_scores(X, labels):
    """
    Compute internal metrics safely.

    Invalid clustering outputs receive:
      - Silhouette = -1
      - CH = 0
      - DB = infinity

    Noise points (-1) are excluded.
    """
    valid = labels != -1

    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"sil": -1.0, "ch": 0.0, "db": np.inf}

    return {
        "sil": float(silhouette_score(X[valid], labels[valid])),
        "ch": float(calinski_harabasz_score(X[valid], labels[valid])),
        "db": float(davies_bouldin_score(X[valid], labels[valid]))
    }


def normalize_higher_better(x, mn, mx):
    """
    Min-max normalization for metrics where larger values are better:
        (x - min) / (max - min + eps)

    No clipping is applied.
    """
    if not np.isfinite(x):
        return 0.0

    return float((x - mn) / (mx - mn + 1e-12))


def normalize_lower_better(x, mn, mx):
    """
    Reverse min-max normalization for metrics where smaller values are better:
        (max - x) / (max - min + eps)

    This is the corrected Davies–Bouldin normalization.
    No clipping is applied.
    """
    if not np.isfinite(x):
        return 0.0

    return float((mx - x) / (mx - mn + 1e-12))


def internal_objective(int_raw, norm_state, return_components=False):
    """
    Internal AutoML objective:
      average of normalized Silhouette, log(CH), and DB components.
    """
    sil_n = normalize_higher_better(
        int_raw["sil"],
        norm_state["sil_min"],
        norm_state["sil_max"]
    )

    ch_n = normalize_higher_better(
        math.log1p(int_raw["ch"]),
        norm_state["lch_min"],
        norm_state["lch_max"]
    )

    db_n = normalize_lower_better(
        int_raw["db"],
        norm_state["db_min"],
        norm_state["db_max"]
    )

    value = (sil_n + ch_n + db_n) / 3.0

    if return_components:
        return value, {
            "sil_n": sil_n,
            "ch_n": ch_n,
            "db_n": db_n
        }

    return value


def external_scores(y_true, labels):
    """
    External agreement metrics used only after final model selection.
    """
    return {
        "ri": float(rand_score(y_true, labels)),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels))
    }


def majority_map(cluster_labels, true_labels):
    """
    Development-set majority-vote cluster-to-class mapping.
    """
    out = {}

    for c in np.unique(cluster_labels):
        members = true_labels[cluster_labels == c]

        if len(members) == 0:
            continue

        out[c] = Counter(members).most_common(1)[0][0]

    return out


# ============================================================
# 10) CLUSTERING RUNNERS
# ============================================================
def run_kmeans(X, k):
    return KMeans(
        n_clusters=int(k),
        n_init=50,
        max_iter=300,
        random_state=RNG
    ).fit(X).labels_


def run_agglo_sklearn(X, k, linkage):
    return AgglomerativeClustering(
        n_clusters=int(k),
        linkage=linkage,
        metric="euclidean"
    ).fit_predict(X)


def run_agglo_centroid_scipy(X, k):
    Z = scipy_linkage(X, method="centroid", metric="euclidean")
    lbl = scipy_fcluster(Z, t=int(k), criterion="maxclust") - 1
    return lbl.astype(int)


def run_spectral(X, k, n_neighbors):
    n_neighbors = min(int(n_neighbors), max(2, X.shape[0] - 1))

    return SpectralClustering(
        n_clusters=int(k),
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=RNG
    ).fit_predict(X)


def run_dbscan(X, eps, min_samples):
    return DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples)
    ).fit(X).labels_


def run_optics(X, min_samples, xi, min_cluster_size):
    return OPTICS(
        min_samples=int(min_samples),
        xi=float(xi),
        min_cluster_size=float(min_cluster_size)
    ).fit(X).labels_


def run_hdbscan(X, min_cluster_size, min_samples):
    if hdbscan is None:
        raise ImportError("hdbscan is not installed.")

    return hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples)
    ).fit(X).labels_


def snn_distance_matrix(X, k_snn=20, metric_knn="euclidean"):
    n = X.shape[0]
    k_snn = max(2, min(int(k_snn), n - 1))

    nn = NearestNeighbors(
        n_neighbors=k_snn,
        metric=metric_knn
    )
    nn.fit(X)

    knn_idx = nn.kneighbors(return_distance=False)

    A = np.zeros((n, n), dtype=np.uint8)
    rows = np.arange(n)[:, None]
    A[rows, knn_idx] = 1

    shared = (A @ A.T).astype(np.float32)
    sim = shared / float(k_snn)
    dist = 1.0 - sim

    np.fill_diagonal(dist, 0.0)
    return dist


def run_snn(X, k_snn, eps_snn, min_samples_snn):
    D = snn_distance_matrix(X, k_snn=k_snn)

    return DBSCAN(
        eps=float(eps_snn),
        min_samples=int(min_samples_snn),
        metric="precomputed"
    ).fit(D).labels_


# ============================================================
# 11) CONFIGURATION FUNCTIONS
# ============================================================
ALG_ID_TO_NAME = {
    0: "kmeans",
    1: "agglo_min",
    2: "agglo_max",
    3: "agglo_avg",
    4: "agglo_centroid",
    5: "spectral",
    6: "dbscan",
    7: "optics",
    8: "snn",
    9: "hdbscan" if hdbscan is not None else "kmeans"
}

AVAILABLE_ALGS = list(ALG_ID_TO_NAME.values())


def decode_cfg(
    alg_id,
    k,
    n_neighbors,
    eps,
    min_samples,
    xi,
    min_cluster_size,
    k_snn,
    eps_snn,
    min_samples_snn,
    hdb_min_cluster_size,
    hdb_min_samples
):
    alg_id = int(round(alg_id))
    alg_id = max(0, min(alg_id, len(AVAILABLE_ALGS) - 1))

    alg = AVAILABLE_ALGS[alg_id]
    cfg = {"alg": alg}

    if alg in [
        "kmeans",
        "agglo_min",
        "agglo_max",
        "agglo_avg",
        "agglo_centroid"
    ]:
        cfg["k"] = int(round(k))

    elif alg == "spectral":
        cfg["k"] = int(round(k))
        cfg["n_neighbors"] = int(round(n_neighbors))

    elif alg == "dbscan":
        cfg["eps"] = float(eps)
        cfg["min_samples"] = int(round(min_samples))

    elif alg == "optics":
        cfg["min_samples"] = int(round(min_samples))
        cfg["xi"] = float(xi)
        cfg["min_cluster_size"] = float(min_cluster_size)

    elif alg == "snn":
        cfg["k_snn"] = int(round(k_snn))
        cfg["eps_snn"] = float(eps_snn)
        cfg["min_samples_snn"] = int(round(min_samples_snn))

    elif alg == "hdbscan":
        cfg["min_cluster_size"] = int(round(hdb_min_cluster_size))
        cfg["min_samples"] = int(round(hdb_min_samples))

    return cfg


def run_cfg_labels(cfg, X):
    a = cfg["alg"]

    if a == "kmeans":
        return run_kmeans(X, cfg["k"])

    if a == "agglo_min":
        return run_agglo_sklearn(X, cfg["k"], linkage="single")

    if a == "agglo_max":
        return run_agglo_sklearn(X, cfg["k"], linkage="complete")

    if a == "agglo_avg":
        return run_agglo_sklearn(X, cfg["k"], linkage="average")

    if a == "agglo_centroid":
        return run_agglo_centroid_scipy(X, cfg["k"])

    if a == "spectral":
        return run_spectral(X, cfg["k"], cfg["n_neighbors"])

    if a == "dbscan":
        return run_dbscan(X, cfg["eps"], cfg["min_samples"])

    if a == "optics":
        return run_optics(
            X,
            cfg["min_samples"],
            cfg["xi"],
            cfg["min_cluster_size"]
        )

    if a == "snn":
        return run_snn(
            X,
            cfg["k_snn"],
            cfg["eps_snn"],
            cfg["min_samples_snn"]
        )

    if a == "hdbscan":
        return run_hdbscan(
            X,
            cfg["min_cluster_size"],
            cfg["min_samples"]
        )

    raise ValueError(f"Unknown algorithm: {a}")


def random_config():
    alg = random.choice(AVAILABLE_ALGS)
    cfg = {"alg": alg}

    if alg in [
        "kmeans",
        "agglo_min",
        "agglo_max",
        "agglo_avg",
        "agglo_centroid"
    ]:
        cfg["k"] = random.randint(2, 10)

    elif alg == "spectral":
        cfg["k"] = random.randint(2, 10)
        cfg["n_neighbors"] = random.randint(5, 50)

    elif alg == "dbscan":
        cfg["eps"] = 10 ** random.uniform(-1.0, 0.7)
        cfg["min_samples"] = random.randint(3, 20)

    elif alg == "optics":
        cfg["min_samples"] = random.randint(3, 20)
        cfg["xi"] = random.uniform(0.01, 0.2)
        cfg["min_cluster_size"] = random.uniform(0.02, 0.2)

    elif alg == "snn":
        cfg["k_snn"] = random.randint(5, 60)
        cfg["eps_snn"] = random.uniform(0.05, 0.8)
        cfg["min_samples_snn"] = random.randint(3, 20)

    elif alg == "hdbscan":
        cfg["min_cluster_size"] = random.randint(5, 80)
        cfg["min_samples"] = random.randint(1, 20)

    return cfg


def robust_minmax(arr, default=(0.0, 1.0)):
    """
    Compute finite min/max; fall back to defaults if too few finite values exist.
    """
    arr = np.array(arr, float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 5:
        return default

    return float(np.min(arr)), float(np.max(arr))


# ============================================================
# 12) WARM-UP PHASE: FIX NORMALIZATION BOUNDS
# ============================================================
WARMUP_TRIALS = 40

warm_int = {
    "sil": [],
    "lch": [],
    "db": []
}

print("\nWarmup random trials using INTERNAL metrics only...")

for _ in range(WARMUP_TRIALS):
    cfg = random_config()

    try:
        lbl = run_cfg_labels(cfg, Z_sub)
        i_sc = safe_internal_scores(Z_sub, lbl)

        warm_int["sil"].append(i_sc["sil"])
        warm_int["lch"].append(math.log1p(i_sc["ch"]))
        warm_int["db"].append(i_sc["db"])

    except Exception:
        continue

sil_min, sil_max = robust_minmax(
    warm_int["sil"],
    default=(-1.0, 1.0)
)

lch_min, lch_max = robust_minmax(
    warm_int["lch"],
    default=(0.0, 12.0)
)

db_min, db_max = robust_minmax(
    warm_int["db"],
    default=(0.0, 10.0)
)

norm_state = {
    "sil_min": sil_min,
    "sil_max": sil_max,
    "lch_min": lch_min,
    "lch_max": lch_max,
    "db_min": db_min,
    "db_max": db_max
}

print("\nInternal normalization ranges fixed after warm-up:")
print(norm_state)


# ============================================================
# 13) BAYESIAN OPTIMIZATION - INTERNAL ONLY
# ============================================================
pbounds = {
    "alg_id": (0, len(AVAILABLE_ALGS) - 1),
    "k": (2, 10),
    "n_neighbors": (5, 50),
    "eps": (0.1, 5.0),
    "min_samples": (3, 20),
    "xi": (0.01, 0.2),
    "min_cluster_size": (0.02, 0.2),
    "k_snn": (5, 60),
    "eps_snn": (0.05, 0.8),
    "min_samples_snn": (3, 20),
    "hdb_min_cluster_size": (5, 80),
    "hdb_min_samples": (1, 20)
}

history = []


def bo_objective(
    alg_id,
    k,
    n_neighbors,
    eps,
    min_samples,
    xi,
    min_cluster_size,
    k_snn,
    eps_snn,
    min_samples_snn,
    hdb_min_cluster_size,
    hdb_min_samples
):
    cfg = decode_cfg(
        alg_id,
        k,
        n_neighbors,
        eps,
        min_samples,
        xi,
        min_cluster_size,
        k_snn,
        eps_snn,
        min_samples_snn,
        hdb_min_cluster_size,
        hdb_min_samples
    )

    try:
        lbl = run_cfg_labels(cfg, Z_sub)
        i_sc = safe_internal_scores(Z_sub, lbl)

        value, normalized_components = internal_objective(
            i_sc,
            norm_state,
            return_components=True
        )

    except Exception:
        i_sc = {"sil": -1.0, "ch": 0.0, "db": np.inf}
        normalized_components = {
            "sil_n": 0.0,
            "ch_n": 0.0,
            "db_n": 0.0
        }
        value = 0.0

    history.append({
        "cfg": cfg,
        "value": value,
        "internal": i_sc,
        "normalized": normalized_components
    })

    return value


bo = BayesianOptimization(
    f=bo_objective,
    pbounds=pbounds,
    random_state=RNG,
    verbose=2
)

bo.maximize(
    init_points=10,
    n_iter=50
)

history_sorted = sorted(
    history,
    key=lambda d: d["value"],
    reverse=True
)

best_entry = history_sorted[0]
best_cfg = best_entry["cfg"]

print("\n================ BEST INTERNAL-ONLY AUTOML CONFIG ================")
print("Best config:", best_cfg)
print("Search objective value:", best_entry["value"])
print("Internal metrics on search subset:", best_entry["internal"])
print("Normalized objective components:", best_entry["normalized"])


# ============================================================
# 14) FINAL FIT + HELD-OUT TEST EVALUATION
# ============================================================
def fit_and_predict(best_cfg, Z_train, y_train, Z_test, y_test):
    """
    Refit the selected configuration on the full development set
    and evaluate on held-out test data.
    """
    labels_train = run_cfg_labels(best_cfg, Z_train)
    alg = best_cfg["alg"]

    if alg == "kmeans":
        est = KMeans(
            n_clusters=best_cfg["k"],
            n_init=50,
            max_iter=300,
            random_state=RNG
        ).fit(Z_train)

        labels_train = est.labels_
        labels_test = est.predict(Z_test)

    else:
        valid_clusters = np.unique(labels_train[labels_train != -1])

        if len(valid_clusters) < 1:
            labels_test = -1 * np.ones(len(Z_test), dtype=int)
        else:
            centers = np.array([
                Z_train[labels_train == c].mean(axis=0)
                for c in valid_clusters
            ])

            nearest = pairwise_distances_argmin(Z_test, centers)
            labels_test = valid_clusters[nearest]

    mapping = majority_map(labels_train, y_train)
    y_pred = np.array([mapping.get(c, -1) for c in labels_test])

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    acc = np.trace(cm) / (cm.sum() + 1e-12)

    int_test = safe_internal_scores(Z_test, labels_test)
    ext_test = external_scores(y_test, labels_test)

    return {
        "acc": float(acc),
        "cm": cm,
        "internal_test": int_test,
        "external_test": ext_test,
        "labels_test": labels_test,
        "y_pred": y_pred
    }


final = fit_and_predict(
    best_cfg,
    Z_full_train,
    y_full_train,
    Zte,
    y_test
)

print("\n================ FINAL TEST RESULT ================")
print("Best config:", best_cfg)
print("Test Accuracy:", final["acc"])
print("Internal test:", final["internal_test"])
print("External test:", final["external_test"])
print("Confusion matrix:\n", final["cm"])


# ============================================================
# 15) VISUALIZATIONS FOR THE BINARY TEST SET
# ============================================================
# Important:
#   - Clustering and all quantitative evaluation are performed
#     in the 16-dimensional UMAP latent space.
#   - The 2D UMAP below is used ONLY for qualitative visualization.

print("\n================ GENERATING BINARY VISUALIZATIONS ================")


# ------------------------------------------------------------
# 15.1) 2D UMAP projection of binary test embeddings
# ------------------------------------------------------------
umap_viz_model = umap.UMAP(
    n_components=2,
    metric="cosine",
    n_neighbors=30,
    min_dist=0.05,
    random_state=RNG
).fit(Xtr)

Zte_2d = umap_viz_model.transform(Xte)

test_cluster_labels = final["labels_test"]

plt.figure(figsize=(7.2, 5.4))

scatter = plt.scatter(
    Zte_2d[:, 0],
    Zte_2d[:, 1],
    c=test_cluster_labels,
    cmap="tab10",
    s=18,
    alpha=0.85,
    edgecolors="none"
)

cbar = plt.colorbar(scatter)
cbar.set_label("Cluster ID")

plt.xlabel("UMAP-2D dimension 1")
plt.ylabel("UMAP-2D dimension 2")
plt.title("UMAP Projection of Binary Test Embeddings by Cluster Assignment")


# Add cluster labels with majority class and purity
for cluster_id in sorted(np.unique(test_cluster_labels)):
    mask = test_cluster_labels == cluster_id

    if np.sum(mask) == 0:
        continue

    x_center = np.median(Zte_2d[mask, 0])
    y_center = np.median(Zte_2d[mask, 1])

    cluster_true_labels = y_test[mask]
    majority_class_id, majority_count = Counter(cluster_true_labels).most_common(1)[0]
    purity = majority_count / len(cluster_true_labels)

    majority_class_name = ID_TO_CLASS.get(
        majority_class_id,
        str(majority_class_id)
    )

    annotation_text = (
        f"C{cluster_id}\n"
        f"{majority_class_name}\n"
        f"({purity:.2f})"
    )

    plt.text(
        x_center,
        y_center,
        annotation_text,
        fontsize=8,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85
        )
    )

plt.tight_layout()

binary_clusters_path = "/content/binary_clusters.png"
plt.savefig(binary_clusters_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", binary_clusters_path)


# ------------------------------------------------------------
# 15.2) Cluster × Class count matrix on the binary test set
# ------------------------------------------------------------
cluster_ids = sorted(np.unique(test_cluster_labels))
true_class_ids = sorted(np.unique(y_test))

count_matrix = np.zeros(
    (len(cluster_ids), len(true_class_ids)),
    dtype=int
)

for i, cluster_id in enumerate(cluster_ids):
    for j, class_id in enumerate(true_class_ids):
        count_matrix[i, j] = np.sum(
            (test_cluster_labels == cluster_id) &
            (y_test == class_id)
        )

class_names_for_plot = [
    ID_TO_CLASS.get(class_id, str(class_id))
    for class_id in true_class_ids
]

plt.figure(figsize=(6.2, 4.8))

im = plt.imshow(
    count_matrix,
    cmap="Blues",
    aspect="auto"
)

cbar = plt.colorbar(im)
cbar.set_label("Count")

plt.xticks(
    ticks=np.arange(len(true_class_ids)),
    labels=class_names_for_plot,
    rotation=30,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(cluster_ids)),
    labels=[str(c) for c in cluster_ids]
)

plt.xlabel("True class")
plt.ylabel("Cluster ID")
plt.title("Cluster × Class Counts on Test Set")


# Annotate cell counts
threshold = count_matrix.max() / 2.0 if count_matrix.size > 0 else 0

for i in range(count_matrix.shape[0]):
    for j in range(count_matrix.shape[1]):
        value = count_matrix[i, j]
        text_color = "white" if value > threshold else "black"

        plt.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color=text_color,
            fontsize=11,
            fontweight="bold"
        )

plt.tight_layout()

binary_counts_path = "/content/binary_counts.png"
plt.savefig(binary_counts_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", binary_counts_path)


# ============================================================
# 16) RESULT SUMMARY TABLE
# ============================================================
summary_df = pd.DataFrame([{
    "Objective": "Internal-only AutoML",
    "Best algorithm": best_cfg["alg"],
    "Best config": str(best_cfg),
    "Search value": best_entry["value"],
    "Search Silhouette": best_entry["internal"]["sil"],
    "Search CH": best_entry["internal"]["ch"],
    "Search DB": best_entry["internal"]["db"],
    "Normalized Silhouette": best_entry["normalized"]["sil_n"],
    "Normalized CH": best_entry["normalized"]["ch_n"],
    "Normalized DB": best_entry["normalized"]["db_n"],
    "Test Accuracy": final["acc"],
    "Test Silhouette": final["internal_test"]["sil"],
    "Test CH": final["internal_test"]["ch"],
    "Test DB": final["internal_test"]["db"],
    "Test RI": final["external_test"]["ri"],
    "Test ARI": final["external_test"]["ari"],
    "Test NMI": final["external_test"]["nmi"]
}])

print("\n================ SUMMARY TABLE ================")
display(summary_df)

summary_df.to_csv(
    "/content/internal_only_automl_results_corrected_db_normalization.csv",
    index=False
)

print(
    "Saved to: "
    "/content/internal_only_automl_results_corrected_db_normalization.csv"
)


# ============================================================
# 17) NORMALIZATION DIAGNOSTICS
# ============================================================
# This block documents:
#   - Whether later BO configurations exceed warm-up-derived bounds
#   - Whether normalized values fall outside [0,1]
#   - No clipping is applied in the objective

diagnostic_rows = []

for item in history:
    norm = item["normalized"]

    diagnostic_rows.append({
        "Algorithm": item["cfg"]["alg"],
        "Objective": item["value"],
        "Sil_norm": norm["sil_n"],
        "CH_norm": norm["ch_n"],
        "DB_norm": norm["db_n"],
        "Sil_outside_0_1": norm["sil_n"] < 0 or norm["sil_n"] > 1,
        "CH_outside_0_1": norm["ch_n"] < 0 or norm["ch_n"] > 1,
        "DB_outside_0_1": norm["db_n"] < 0 or norm["db_n"] > 1
    })

normalization_diagnostics_df = pd.DataFrame(diagnostic_rows)

print("\n================ NORMALIZATION DIAGNOSTICS ================")
display(normalization_diagnostics_df.head(20))

outside_summary = pd.DataFrame([{
    "Total BO Evaluations": len(normalization_diagnostics_df),
    "Silhouette outside [0,1]": int(
        normalization_diagnostics_df["Sil_outside_0_1"].sum()
    ),
    "CH outside [0,1]": int(
        normalization_diagnostics_df["CH_outside_0_1"].sum()
    ),
    "DB outside [0,1]": int(
        normalization_diagnostics_df["DB_outside_0_1"].sum()
    )
}])

print("\n================ OUT-OF-RANGE NORMALIZATION SUMMARY ================")
display(outside_summary)

normalization_diagnostics_df.to_csv(
    "/content/internal_only_automl_normalization_diagnostics.csv",
    index=False
)

outside_summary.to_csv(
    "/content/internal_only_automl_normalization_out_of_range_summary.csv",
    index=False
)

print("\nSaved:")
print("/content/internal_only_automl_normalization_diagnostics.csv")
print("/content/internal_only_automl_normalization_out_of_range_summary.csv")
