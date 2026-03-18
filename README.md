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

### How to Run Each Code File 
Follow the steps to access the Kaggle dataset through Google Colab, and the provided code notebooks should run without any errors

### Data Preprocessing Steps How To Train and Run Each Model, and Expected OutPuts 
Each code notebook (with code on respective models) should be able to run in sequence, completing any data cleaning and preprocessing without throwing any errors, upon following the steps to access the data through Google Colab.


