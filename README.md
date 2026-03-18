# Malignancy Detection in Skin Cancer Image Data
## A Multimodal Deep Learning Approach
### CS273P Final Project-UC Irvine, Donald Bren School of Information and Computer Sciences

**Team:** Ana Nguyen · Akshaya Bharadhwaj · Shrinidhi Prasanna


## Project Overview
This project addresses malignant skin lesion detection using the ISIC 2024 Challenge dataset,
which contains 401,059 dermoscopic images captured via 3D Total Body Photography (TBP)
alongside rich clinical metadata. We frame this as a binary classification problem
(Benign vs. Malignant) and compare seven models of increasing complexity across three
modality configurations: tabular only, image only, and combined tabular + image.

## Repository Structure
```
├── README.md
├── notebooks/
│   ├── isic2024_xgboost_cnn_multimodal_classification.ipynb
│   │     → Akshaya: EDA, XGBoost, EfficientNet-B0 CNN,
│   │                CNN + Tabular combined model
│   ├── [Ana_notebook].ipynb
│   │     → Ana: EDA, LSTM, Additional Models
│   └── isic2024_logreg_xgboost_stacked_binary_classification.ipynb
│         → Shrinidhi: Stacked XGBoost, CNN and Combined Binary Classification Models
└── report/
    └── CS273P_Final_Project_Report.pdf
```

## Dataset and Setup Instructions <br>

### Where and How to Download
The data can be found at Kaggle: [Skin Cancer Detection](https://www.kaggle.com/competitions/isic-2024-challenge/data) <br>
Navigate to the competition page: https://www.kaggle.com/competitions/isic-2024-challenge/data <br>
Accept the competition rules to gain data access. <br>
<br>
Because of the sheer size of the data, the code notebooks largely used Google Colab, accessing the data om Kaggle using a generated API Token.

### Setup Intsructions:
### Option A: Google Colab + Kaggle (Recommended)

All notebooks were developed and tested on Google Colab using Kaggle's dataset API.
This is the recommended environment.

**Step 1: Set up Kaggle API credentials**
1. Go to https://www.kaggle.com/settings → Account → API
2. Click **"Create New Legacy Key"** — downloads `kaggle.json` to your machine
3. Store this file on your device as you will upload it when prompted by the notebook

**Step 2: Enable GPU (required for CNN notebooks)**
1. In Colab: **Runtime → Change runtime type → T4 GPU**
2. Click Save

**Step 3: Run the notebook**
1. Open the desired notebook in Colab
2. Run the first setup cell — upload your `kaggle.json` when prompted
3. The notebook will automatically download and extract the dataset (~50GB)
4. Run all remaining cells in order from top to bottom

### Option B: Local Setup

**System requirements**
- Python 3.9+
- CUDA-compatible GPU strongly recommended for CNN notebooks
- ~60GB free disk space

**Install dependencies**
```bash
pip install torch torchvision
pip install scikit-learn xgboost imbalanced-learn
pip install pandas numpy matplotlib seaborn
pip install pillow kaggle
```

**Download dataset locally**
```bash
# Place kaggle.json at ~/.kaggle/kaggle.json first
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and extract
kaggle competitions download -c isic-2024-challenge
unzip isic-2024-challenge.zip
```

The following files will be downloaded on google colab or your local system:
   - `train-metadata.csv` - clinical metadata for all training lesions (55 features)
   - `train-image/image/` - folder of JPEG lesion images named by `isic_id`
   - `test-metadata.csv` - test set metadata (no labels, not used in this project)


### Dataset Structure
```
train-metadata.csv: 401,059 rows, 55 columns, includes binary target label
train-image/image/: 401,059 JPEG images (e.g. ISIC_0015670.jpg)
test-metadata.csv: 3 rows, no labels (competition holdout-not used)
```

## Notebooks Guide

### Akshaya's Notebook — `isic2024_xgboost_cnn_multimodal_classification.ipynb`

Contains EDA and three modeling approaches. **GPU required for CNN sections.**

**Section 1. Exploratory Data Analysis**
- Class imbalance visualization (linear and log scale)
- Age, sex, anatomical site analysis
- Lesion size distributions by class
- TBP feature correlation heatmap and top feature distributions
- Image channel intensity and contrast analysis
- Patient-level lesion count distribution

**Section 2. Data Preparation**
- Label encoding (Indeterminate merged into Benign)
- 24 tabular feature selection based on EDA
- Patient-aware 70/15/15 train/val/test split via GroupShuffleSplit
- StandardScaler fit on train set only

**Section 3. Model 1: XGBoost (Tabular Baseline)**
- 24 clinical metadata features, `scale_pos_weight` for imbalance
- Expected Val ROC-AUC: ~0.920
- Runtime: ~5 minutes (CPU)
- Outputs: feature importance plot, ROC curve, confusion matrices, PR curve

**Section 4. Model 2: EfficientNet-B0 (Image Only)**
- Pretrained EfficientNet-B0, two-phase training (frozen 5 epochs → full fine-tune 5 epochs)
- Malignant oversampling 10x, weighted BCE loss
- Expected Val ROC-AUC: ~0.858, Malignant Recall: ~0.93
- Runtime: ~30–50 minutes (GPU T4)
- Outputs: training curves, ROC curve, PR curve, clinical tradeoff plot, confusion matrices

**Section 5. Model 3: CNN + Tabular Combined**
- Custom multimodal architecture: EfficientNet image branch (1280-dim) +
  tabular MLP branch (24→128→64-dim) → concatenated → classifier MLP
- Same two-phase training strategy as Model 2
- Expected Val ROC-AUC: ~0.858, Malignant Recall: ~0.93
- Runtime: ~50–80 minutes (GPU T4)
- Outputs: training curves, ROC comparison vs CNN-only, confusion matrices

**Section 6. Final Comparison**
- ROC curves for all three models on the same held-out test set
- AUC bar chart, summary metrics table

---

### Ana's Notebook — `isic2024_logreg_LSTM.ipynb`
Contains logistic regression with odds ratio analysis and two LSTM architectures.

**Section 1. Imports, Config & Feature Engineering**
- Full library setup: pandas, numpy, sklearn, PyTorch, torchvision, h5py, scipy
- 15 TBP measurement columns selected as numeric features
- 2 demographic features: age_approx, clin_size_long_diam_mm
- 3 categorical features: sex, anatom_site_general, tbp_tile_type
- 4 engineered interaction features:
  - feat_border_x_color (border × colour normalisation)
  - feat_age_x_size (age × lesion diameter)
  - feat_asymm_ratio (border irregularity / symmetry)
  - feat_compact_x_color (area-perimeter ratio × colour normalisation)
- Total: 24 features (20 numeric + 4 engineered + categorical OHE expansion)
- Positive class weight computed as n_neg / n_pos ≈ 1,019×

**Section 2. Stratified 5-Fold CV Framework**
- 80/20 stratified train_test_split on full 401,059-row dataset (held-out 20% = ~80,000 rows)
- StratifiedKFold(n_splits=5) applied to development set only
- Four evaluation metrics computed per fold:
  - AUROC (full curve)
  - Partial AUROC (specificity ≥ 80% zone, clinically relevant region)
  - Average Precision (PR-AUC)
  - Sensitivity at 80% specificity
- Fold balance verification printed and visualised

**Section 3. Logistic Regression — 5-Fold Stratified CV**
- Pipeline: median imputer → StandardScaler (numeric) + most-frequent imputer → OneHotEncoder (categorical) → LogisticRegression(C=1.0, solver=lbfgs, class_weight=balanced)
- Expected Val ROC-AUC: ~0.897 ± 0.010
- Fold stability: AUC range 0.882–0.909 across all five folds
- Runtime: ~2–5 minutes (CPU)
- Outputs: 4-panel diagnostic figure (ROC curves per fold, metric stability chart, hold-out confusion matrix at threshold 0.995, threshold vs sensitivity/specificity/precision/F1 curve)

**Section 4. Odds Ratio Forest Plot**
- Logistic regression coefficients exponentiated to odds ratios
- 95% confidence intervals via vectorised bootstrap (200 resamples, n=15,000 subsample) — runs in under 60 seconds
- Top 25 features plotted on log-scale x-axis with linear OR magnitude bar chart
- Diamond markers = CI excludes 1.0 (statistically significant); circle = not significant
- Key findings: tbp_lv_norm_border OR=7,526 (extreme, likely quasi-complete separation), feat_compact_x_color OR=1,159, tbp_lv_area_perim_ratio OR=0.003 (strongly protective), tbp_lv_symm_2axis OR=0.024 (symmetric lesions protective)
- ABCDE criteria independently recovered: border (B), colour (C), diameter (D), asymmetry (A) all appear in top features

**Section 5. LSTM Models — Tabular and CNN-Image**
- 80/20 stratified holdout split before any fold training (holdout: 4,000 rows, 4 malignant)
- Tabular LSTM: 35 features → sequence of length 35 (1 feature per timestep) → 2-layer LSTM (hidden=64, dropout=0.3) → BatchNorm → Dropout(0.3) → Linear(64→1) → sigmoid
- CNN-Image LSTM: 64×64 RGB image → 64 pixel rows each compressed by 2× Conv1d to 32-dim vector → bidirectional 2-layer LSTM (hidden=128) → BatchNorm → Dropout(0.3) → Linear(256→1) → sigmoid
- Training: Adam (lr=1e-3, weight_decay=1e-4), 12 epochs, ReduceLROnPlateau (×0.5, patience=2), gradient clipping at norm 1.0
- Imbalance: WeightedRandomSampler with pos_weight ≈ 1,019, BCELoss; image branch adds H/V flip and colour jitter augmentation
- Threshold: Youden's J statistic on OOF predictions (not 0.5); also evaluated at clinical threshold 0.30
- Expected Tabular LSTM Val ROC-AUC: ~0.666 ± 0.042; Image LSTM: ~0.803 ± 0.070
- Hold-out Tabular LSTM ROC-AUC: 0.661, PR-AUC: 0.005; Image LSTM ROC-AUC: 0.814, PR-AUC: 0.061
- Runtime: ~10–20 minutes (GPU T4)
- Outputs: per-fold AUC tables, overfitting diagnostic (train vs val AUC per epoch), threshold optimisation printout, 4-threshold confusion matrices with clinical interpretation, classification reports

---

### Shrinidhi's Notebook — `isic2024_logreg_xgboost_stacked_binary_classification.ipynb`
Contains 3 approaches: Logistic Regression, XGBoost, and Stacked Model. GPU recommended for feature extraction sections.

**Section 1. Data Loading and EDA**
- Kaggle dataset download via kagglehub API
- Visualization of first 25 lesion images
- Feature documentation and column descriptions
- Missing value heatmap and analysis
- Correlation matrix of all numeric features
- Class imbalance identification and discussion of handling strategies

**Section 2. Data Splitting**
- Stratified 70/15/15 train/val/test split using `train_test_split`
  with `stratify=target` to preserve class ratios across splits

**Section 3. Model 1: Logistic Regression (Tabular Only)**
- Full tabular dataset (280K rows, 34 numeric + 4 categorical features)
- Pipeline: SimpleImputer → StandardScaler (numeric),
  SimpleImputer → OneHotEncoder (categorical)
- Class weight set to `balanced` to handle imbalance
- 5-fold stratified cross-validation
- Expected CV ROC-AUC: ~0.893, CV PR-AUC: ~0.029
- Outputs: feature coefficient plot, confusion matrix, ROC curve, PR curve

**Section 4. Model 2: Logistic Regression (Tabular + CNN Embeddings)**
- EfficientNet-B0 used to extract 1280-dim image embeddings from a
  balanced subsample of 10K rows (RAM constraints prevented full dataset)
- Embeddings merged with tabular features → 1335 total features
- Same Logistic Regression pipeline retrained on combined features
- Embeddings saved to Google Drive as `.parquet` to avoid recomputation
- Expected Val ROC-AUC: ~0.818, Val PR-AUC: ~0.206
- Outputs: confusion matrix, ROC curve, PR curve (train and validation)

**Section 5. Model 3: XGBoost (Tabular Only)**
- Full tabular training set (280K rows)
- `scale_pos_weight` set to ~1020:1 to handle class imbalance
- 500 estimators, max depth 6, learning rate 0.05
- 5-fold stratified cross-validation
- Expected CV ROC-AUC: ~0.924, Val ROC-AUC: ~0.920
- Outputs: feature importance plot, confusion matrix, ROC curve, PR curve

**Section 6. Model 4: XGBoost (Tabular + CNN Embeddings)**
- Same subsampled embedded dataset as Section 4
- XGBoost retrained with 1314 numeric + 4 categorical features
- Expected CV ROC-AUC: ~0.941, Val ROC-AUC: ~0.949, Val PR-AUC: ~0.552
- Outputs: confusion matrix, ROC curve, PR curve (train and validation)

**Section 7. Model 5: Stacked XGBoost (Meta-Model)**
- Generates out-of-fold (OOF) predictions from all 4 base models above
- OOF predictions assembled into a meta-feature dataframe
  (4 columns: logreg_tab, xgb_tab, logreg_cnn, xgb_cnn)
- XGBoost meta-model trained on these stacked predictions
- Learns optimal weighting of base model outputs
- Expected Val ROC-AUC: ~0.922, Malignant Recall: ~0.83
- Outputs: confusion matrix, ROC curve, PR curve

**Note:** 
- Sections 4 and 6 require Google Drive mounting to read/write
- saved embedding parquet files. Runtime may crash on full data due to RAM
- limits: the notebook uses a 10K subsample for embedding-based models.
---

## Data Preprocessing Summary

The following steps are implemented across the notebooks:

| Step | Details |
|---|---|
| Label encoding | Indeterminate → Benign (0), Malignant → 1 |
| Categorical encoding | `sex`, `anatom_site_general` label-encoded; NaN → 'unknown' |
| Missing values | Rows with NaN in selected features dropped |
| Feature scaling | StandardScaler fit on train only, applied to val/test |
| Data splitting | GroupShuffleSplit by `patient_id` — 70/15/15 train/val/test |
| Class imbalance | Malignant oversampled 10x; benign downsampled to 5K (CNN models) |
| Loss weighting | BCEWithLogitsLoss `pos_weight` = benign:malignant ratio |
| XGBoost imbalance | `scale_pos_weight` = negative:positive class ratio (~1000:1) |
| Image resizing | All images resized to 224×224 |
| Normalization | ImageNet mean/std: [0.485,0.456,0.406] / [0.229,0.224,0.225] |
| Augmentation | Random flips, rotation ±20°, color jitter (train only) |
| Image embedding | EfficientNet-B0 used to extract 1280-dim features for LR/XGBoost variants |

---

## How to Reproduce Key Results

All random seeds are fixed (`random_state=42`). Run each notebook end-to-end
from top to bottom. Expected results on the validation/test set:

| Model | Val ROC-AUC | Val PR-AUC | Malignant Recall | Benign Recall |
|---|---|---|---|---|
| LogReg (Tabular) | 0.9196 | 0.0256 | 0.85 | 0.82 |
| LogReg (Tabular + Image) | 0.8183 | 0.2064 | 0.39 | 0.96 |
| XGBoost (Tabular) | 0.9202 | 0.0475 | 0.12 | ~1.00 |
| XGBoost (Tabular + Image) | 0.9485 | 0.5515 | 0.32 | ~1.00 |
| EfficientNet-B0 (Image) | 0.8582 | — | 0.93 | 0.51 |
| CNN + Tabular Combined | 0.8580 | — | 0.93 | 0.80 |
| Stacked XGBoost | 0.9222 | 0.0433 | 0.83 | 0.89 |

- Minor variation (~±0.005 AUC) may occur due to GPU non-determinism in PyTorch.

---

## Expected Outputs

Running the full modeling notebooks produces:

**EDA:**
-  Class distribution plots (linear + log scale)
-  Age distribution and boxplots by target
-  Sex and anatomical site analysis plots
-  Lesion size distributions by class
-  TBP feature correlation heatmap
-  Top feature distribution plots (benign vs malignant)
-  Patient-level lesion count histograms

**Models:**
-  XGBoost feature importance bar chart
-  Training loss and AUC curves (CNN models)
-  ROC curves per model + final comparison
-  Precision-Recall curves per model
-  Clinical tradeoff plot (recall vs false positive count)
-  Confusion matrices at multiple thresholds
-  Summary metrics table across all models

---

## Team Contributions

| Member | Key Contributions |
|---|---|
| Akshaya Bharadhwaj | EDA, XGBoost , EfficientNet-B0 CNN, CNN + Tabular combined model, report writing |
| Ana Nguyen |EDA, Logistic Regression Baseline approach, combines LSTM, Image + Tabular data model, report writing |
| Shrinidhi Prasanna | Baseline Logistic Regression, XGBoost, and Stacked Model and combined models, report writing |

---

## References

- ISIC 2024 Challenge: https://www.kaggle.com/competitions/isic-2024-challenge
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- XGBoost: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016
- Debelee, T.G. "Skin Lesion Classification and Detection Using Machine Learning
  Techniques: A Systematic Review." Diagnostics, 2023
- Brinker et al. "Diagnostic Performance of AI for Histologic Melanoma Recognition
  Compared to 18 International Expert Pathologists." JAAD, 2022

---

## License

This project is for academic purposes only as part of CS273P at UC Irvine.
The ISIC 2024 dataset is subject to its own competition terms and conditions.
