# Internal-Only AutoML for Binary Chest X-Ray Clustering

A Google Colab implementation of an **internal-metric-only AutoML pipeline** for unsupervised clustering of binary chest X-ray images (*Normal* and *Pneumonia*).

The pipeline extracts frozen self-supervised DINO features, applies train-set standardization and L2 normalization, reduces dimensionality with UMAP, searches clustering algorithms and hyperparameters with Bayesian optimization, and evaluates the selected configuration on a held-out test set.

> **Research-use notice.** This repository is intended for research and reproducibility only. It is not a clinical decision-support system and must not be used for diagnosis or patient-care decisions.

---

## Highlights

- Uses a frozen **DINO ViT-S/16** feature extractor through `timm`.
- Fits preprocessing and UMAP using the training split only.
- Uses a fixed 800-sample subset from the combined training and validation data for the AutoML search.
- Searches multiple clustering families:
  - K-means
  - Agglomerative clustering: single, complete, average, and centroid linkage
  - Spectral clustering
  - DBSCAN
  - OPTICS
  - Shared-nearest-neighbor (SNN) clustering with DBSCAN
  - HDBSCAN, when the package is available
- Optimizes an **internal-only objective** built from normalized Silhouette score, log-transformed Calinski-Harabasz score, and reverse-normalized Davies-Bouldin score.
- Uses 40 warm-up random trials to establish normalization ranges before Bayesian optimization.
- Uses no labels for model/parameter selection. Ground-truth labels are used only after selection for held-out evaluation and cluster-to-class interpretation.
- Provides a common out-of-sample strategy for non-predictive clustering methods by assigning test samples to the nearest non-noise development-set cluster centroid.
- Creates manuscript-ready qualitative visualizations and CSV result summaries.

---

## Repository Contents

```text
.
├── automl_clustering_binary_chest_xray_colab.py   # Main Google Colab implementation
├── README.md                                      # Project documentation
├── requirements.txt                               # Python dependencies
└── .gitignore                                     # Files/folders excluded from version control
```

> The repository intentionally does **not** include the chest X-ray dataset, patient data, trained model weights, or large generated outputs.

---

## Dataset Structure

The notebook expects a **pre-split binary dataset** with this exact directory layout:

```text
YOUR_DATASET_ROOT/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Supported image formats are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.

The code expects the two class names `NORMAL` and `PNEUMONIA`. If your folder names differ, update the following settings in the code:

```python
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}
ID_TO_CLASS = {0: "Normal", 1: "Pneumonia"}
```

---

## Requirements

### Recommended environment

- Google Colab with a GPU runtime recommended for DINO feature extraction.
- Google Drive, because the provided implementation mounts Drive and copies the dataset to local Colab storage before loading images.
- Python packages listed in `requirements.txt`.

### Install packages

The first code block installs the required packages in Colab:

```python
!pip -q install timm umap-learn hdbscan bayesian-optimization
```

For a non-Colab environment, install dependencies with:

```bash
pip install -r requirements.txt
```

> The supplied implementation is written for **Google Colab**. For fully local execution, replace the Google Drive mounting section with a local dataset path and replace notebook-only commands such as `!pip` with standard environment setup.

---

## How to Run in Google Colab

1. Upload `automl_clustering_binary_chest_xray_colab.py` to Google Drive, or copy its content into a new Google Colab notebook.
2. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU), if available.
3. Organize your dataset in Google Drive using the folder structure above.
4. Update the dataset path in the configuration section:

```python
DRIVE_ROOT = "/content/drive/MyDrive/YOUR_DATASET_FOLDER"
LOCAL_ROOT = "/content/Chest_XRay_updated_local"
```

5. Run the notebook from top to bottom.
6. Review the selected configuration, held-out test metrics, figures, and CSV summaries saved under `/content/`.

---

## Pipeline Overview

### 1. Dataset indexing and robust file access

The code checks the expected train/validation/test layout, verifies class folders, counts images, and copies the dataset from Google Drive to local Colab storage. This reduces image-loading problems that can occur when a DataLoader reads directly from Drive.

### 2. DINO feature extraction

A pretrained frozen `vit_small_patch16_224.dino` model is created using `timm`. Images are converted to grayscale, expanded to three channels, resized to 224 × 224 pixels, normalized using ImageNet statistics, and passed through the DINO model to obtain CLS-token feature vectors.

### 3. Feature normalization

A `StandardScaler` is fitted on training features only. The same scaler transforms validation and test features, after which each feature vector is L2-normalized.

### 4. UMAP dimensionality reduction

A 16-dimensional UMAP model is fitted on the normalized training features using cosine distance, `n_neighbors=30`, `min_dist=0.05`, and a fixed random seed. The fitted model then transforms validation and test features.

### 5. Internal-only AutoML search

The development pool is formed by combining training and validation embeddings. A random 800-sample subset is used for hyperparameter search. The search uses:

- 40 random warm-up trials to set fixed normalization bounds;
- 10 Bayesian-optimization initialization trials; and
- 50 Bayesian-optimization iterations.

The optimization objective is the mean of three normalized internal validity components:

- Silhouette score: higher is better;
- `log1p(Calinski-Harabasz score)`: higher is better;
- Davies-Bouldin score: lower is better, so reverse min-max normalization is applied.

No clipping is used after normalization. Diagnostic tables record whether later trials fall outside the warm-up ranges.

### 6. Final fit and held-out evaluation

The selected configuration is re-fitted on the full development set. K-means uses its native prediction rule. For other clustering methods, test samples are assigned to their nearest valid development-set cluster centroid. A development-set majority-vote mapping converts cluster assignments to class labels for reporting test accuracy.

Ground-truth test labels are also used to calculate external agreement metrics:

- Rand Index (RI)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

---

## Outputs

The implementation saves the following files in the Colab `/content/` directory:

| File | Description |
|---|---|
| `binary_clusters.png` | Two-dimensional UMAP visualization of the test embeddings colored by assigned cluster. |
| `binary_counts.png` | Cluster × true-class count matrix for the binary test set. |
| `internal_only_automl_results_corrected_db_normalization.csv` | Best configuration and search/test metric summary. |
| `internal_only_automl_normalization_diagnostics.csv` | Per-trial normalized objective components and out-of-range flags. |
| `internal_only_automl_normalization_out_of_range_summary.csv` | Summary count of normalized components outside the `[0, 1]` range. |

---

## Reproducibility Settings

The default random seed is:

```python
RNG = 42
```

The code applies this seed to Python's `random`, NumPy, and PyTorch. Although this improves reproducibility, exact results can still vary across hardware, CUDA versions, package versions, and non-deterministic GPU operations.

---

## Important Data and Privacy Notes

- Do **not** upload chest X-ray images, patient metadata, passwords, API keys, private Google Drive links, or institutional credentials to GitHub.
- Share the dataset only if you have the legal and ethical right to do so.
- If the source dataset has its own license or access restrictions, provide the official source link and access instructions instead of redistributing its files.
- Before making the repository public, replace any personal/local path with a generic placeholder such as `YOUR_DATASET_FOLDER`.

---

## Citation

If you use or adapt this code, please cite the associated manuscript once its final bibliographic information is available.

```bibtex
@article{REPLACE_WITH_FINAL_CITATION,
  title   = {REPLACE WITH FINAL ARTICLE TITLE},
  author  = {REPLACE WITH FINAL AUTHOR LIST},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {REPLACE WITH FINAL DOI}
}
```

---

## License

Add a license only after confirming the preferred licensing terms with all code contributors and your supervisor. For an openly reusable research-code repository, the MIT License is commonly used, but the final choice should match your project and institutional requirements.

---

## Contact

**Salma Ouhsousou**  
Data Science Master’s Student, Ondokuz Mayıs University  
GitHub: `ADD_YOUR_GITHUB_PROFILE_LINK`  
ORCID: `ADD_YOUR_ORCID_LINK`
