# FreshGuard AI – Intelligent Food Freshness Detection and Classification System

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.9+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com)

**Master of Computer Applications (MCA) Master Project**  
**Student Name:** Hariom Gupta  
**UID:** 25MCA20022  
**Domain:** Machine Learning, Computer Vision & Relational Database Architecture  

---

## 1. Project Introduction

**FreshGuard AI** is an intelligent, full-stack machine learning web application engineered to detect and classify the apparent freshness of fruits and vegetables (such as Apples, Bananas, and Oranges) from uploaded digital imagery.

By utilizing feature-engineered classical algorithms—specifically **Support Vector Machines (SVM)** and **K-Nearest Neighbors (KNN)** with Histogram of Oriented Gradients (HOG) and HSV color moments—the system provides fast, objective, non-contact optical inspection. All inspection records, confidence scores, and diagnostic metadata are persistently logged in a **MySQL 8.0** relational database.

> **Crucial Rule:** In strict accordance with the project specification, **MySQL 8.0** is the sole database engine used. SQLite is strictly prohibited and not used anywhere in this codebase.

---

## 2. Problem Statement

Food waste is a major global issue. The Food and Agriculture Organization (FAO) reports that roughly one-third of all food produced is lost or discarded, with fresh produce suffering the highest rates of post-harvest loss due to microbial spoilage and physical damage. Manual quality inspection in distribution centers and grocery retail is:
- **Subjective:** Heavily reliant on worker experience and perception.
- **Labor-Intensive & Slow:** Cannot scale to high-throughput sorting conveyor lines.
- **Inconsistent:** Human fatigue leads to misgrading and premature disposal.

FreshGuard AI solves this challenge by delivering an automated, non-invasive image-based freshness classifier that instantly labels produce as **FRESH**, **ROTTEN**, or **UNCERTAIN**, logging every audit trail into MySQL.

---

## 3. Aim and Objectives

### Aim
To develop, implement, and benchmark an intelligent food freshness detection system using deep learning and machine learning techniques to classify produce images and maintain a robust audit log in MySQL.

### Core Objectives
1. **Automated Dataset Ingestion:** Download, verify, and split genuine fruit images into stratified training, validation, and testing partitions with no data leakage.
2. **Deep Learning (CNN):** Architect a multi-layer Convolutional Neural Network with data augmentation, Batch Normalization, and Dropout.
3. **Classical ML (SVM & KNN):** Extract 1818-dimensional feature vectors combining Dalal-Triggs HOG descriptors, HSV color histograms, and RGB channel moments.
4. **Empirical Benchmarking:** Evaluate all three models on the exact same held-out test dataset (161 images), generating real accuracy, precision, recall, F1-scores, and confusion matrices without artificial values.
5. **MySQL 8.0 Integration:** Implement a data access repository for storing every classification record, supporting parameterized search, filtering, pagination, single-record deletion, and table clearing.
6. **Responsive Web Dashboard:** Develop a user-friendly UI following a deep forest green, fresh lime, and warm orange design language.

---

## 4. Features

- **Multi-Model Inference:** Select between CNN, SVM (HOG), and KNN (Texture/Color) for any uploaded image.
- **Drag-and-Drop Upload:** Client-side drag-and-drop zone with instant image preview and format/size validation.
- **Calibrated Uncertainty Thresholding:** If confidence falls below 60%, the system flags the result as `UNCERTAIN` and advises clearer lighting.
- **Analytics Dashboard:** Visual indicators showing total scans, fresh vs. rotten distribution, average confidence, and model performance.
- **Prediction History (MySQL):** Full audit table with search by category, filter by freshness status or model name, and pagination.
- **Record Management:** Delete individual records or truncate history table with modal confirmations.
- **Model Comparison Suite:** Dynamic display of accuracy bar charts, multi-metric charts, confusion matrix heatmaps, and CNN loss/accuracy history curves.
- **REST Health API (`/health`):** JSON endpoint monitoring Flask server status, model memory cache, and live MySQL connection.

---

## 5. Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Backend** | Python 3.14+, Flask 3.1 | Microframework handling routing, templates, and inference |
| **Deep Learning** | TensorFlow 2.22 / Keras 3.16 | CNN training, layers, data augmentation, model saving |
| **Machine Learning** | Scikit-learn 1.9 | SVM (RBF kernel), KNN (k=3), StandardScaler, evaluation |
| **Computer Vision** | OpenCV 5.0, Pillow 12.3 | Image resizing, color conversions, HOG & HSV extraction |
| **Data & Visualization**| NumPy 2.5, Pandas 3.0, Matplotlib, Seaborn | Data handling, confusion matrices, comparison bar charts |
| **Database** | MySQL Server 8.0, `mysql-connector-python` | Sole relational database engine; parameterized queries |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5.3, Bootstrap Icons | Responsive UI with custom cards, badges, and animations |

---

## 6. Dataset Information

The system utilizes the genuine **Fruits Fresh and Rotten for Classification** dataset (mirrored from Kaggle/HuggingFace):
- **Classes (6 categories):**
  1. `fresh_apple` (159 total)
  2. `fresh_banana` (147 total)
  3. `fresh_orange` (138 total)
  4. `rotten_apple` (230 total)
  5. `rotten_banana` (224 total)
  6. `rotten_orange` (160 total)
- **Total Images:** 1,058 genuine images
- **Splits (Stratified without Data Leakage):**
  - **Training Set (70%):** 736 images
  - **Validation Set (15%):** 161 images
  - **Held-Out Test Set (15%):** 161 images (Shared identically across all 3 models)

---

## 7. Project Structure

```text
FreshGuard/
│
├── app.py                     # Main Flask web application
├── config.py                  # Environment and application configuration
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── .env                       # Active MySQL credentials & environment configuration
├── .env.example               # Template environment configuration
├── .gitignore                 # Version control exclusions
│
├── database/
│   ├── __init__.py
│   ├── db_connection.py       # MySQL connection pool & health checks
│   ├── schema.sql             # SQL schema for freshguard_db
│   └── prediction_repository.py # Parameterized queries, CRUD operations, stats
│
├── dataset/                   # Automatically managed dataset splits
│   ├── raw/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── dataset_summary.csv
│
├── models/
│   ├── cnn_model/
│   │   └── freshguard_cnn.keras # Saved Keras CNN model
│   ├── svm_model.pkl          # Serialized SVM model + StandardScaler
│   ├── knn_model.pkl          # Serialized KNN model + StandardScaler
│   └── model_metadata.json    # Empirical test metrics
│
├── training/
│   ├── download_dataset.py    # Multithreaded dataset fetcher & splitter
│   ├── preprocess.py          # HOG & Color feature extraction, CNN normalization
│   ├── train_cnn.py           # CNN architecture & training pipeline
│   ├── train_svm.py           # SVM training with HOG/color features
│   ├── train_knn.py           # KNN k-tuning & training pipeline
│   └── evaluate.py            # Unified evaluation, confusion matrices, charts
│
├── static/
│   ├── css/
│   │   └── style.css          # Custom deep forest green & lime design system
│   ├── js/
│   │   └── script.js          # Drag-and-drop, client-side validation
│   └── uploads/               # Securely stored user-uploaded test images
│
├── templates/
│   ├── base.html              # Master layout with navigation & footer
│   ├── index.html             # Landing page
│   ├── dashboard.html         # Live analytics dashboard
│   ├── upload.html            # Drag-and-drop upload interface
│   ├── result.html            # Detailed classification result page
│   ├── history.html           # Prediction history with MySQL search & filter
│   ├── comparison.html        # Model comparison dashboard
│   └── about.html             # MCA student and project details
│
└── reports/
    ├── confusion_matrix/      # Generated confusion matrices (CNN, SVM, KNN)
    ├── training_history/      # CNN accuracy & loss curves
    ├── accuracy_comparison.png# Comparative bar chart
    ├── metrics_comparison.png # Grouped precision/recall/F1 chart
    └── classification_report.txt # Raw classification report output
```

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- MySQL Server 8.0 or compatible
- Google Chrome or any modern browser

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "C:\Users\harsh\.gemini\antigravity\scratch\FreshGuard"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. MySQL Database Configuration

**Important:** SQLite is strictly prohibited. You must have MySQL 8.0 installed and running.

### Database Setup
1. Log into your MySQL CLI or MySQL Workbench:
   ```bash
   mysql -u root -p
   ```
2. Execute the database creation script:
   ```sql
   CREATE DATABASE IF NOT EXISTS freshguard_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   USE freshguard_db;
   ```
3. Create the `prediction_history` table:
   ```sql
   CREATE TABLE IF NOT EXISTS prediction_history (
       id INT AUTO_INCREMENT PRIMARY KEY,
       image_filename VARCHAR(255) NOT NULL,
       predicted_category VARCHAR(100) NOT NULL,
       freshness_status VARCHAR(50) NOT NULL,
       confidence_score FLOAT,
       model_name VARCHAR(100) NOT NULL,
       prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       INDEX idx_freshness_status (freshness_status),
       INDEX idx_model_name (model_name),
       INDEX idx_prediction_date (prediction_date)
   ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
   ```

### Step 3: Configure Environment Variables
Edit `.env` to match your MySQL server host, port, and credentials:
```env
DB_HOST=127.0.0.1
DB_PORT=3307
DB_USER=root
DB_PASSWORD=
DB_NAME=freshguard_db
SECRET_KEY=freshguard_mca_2026_hariom_secret_key
CONFIDENCE_THRESHOLD=0.60
```
*(If running on the default MySQL service, set `DB_PORT=3306` and enter your password).*

---

## 10. Model Training Pipeline

Execute the following commands in sequence to acquire data and train all models:

```bash
# 1. Download and automatically split the dataset
python training/download_dataset.py

# 2. Train Convolutional Neural Network (CNN)
python training/train_cnn.py

# 3. Train Support Vector Machine (SVM)
python training/train_svm.py

# 4. Train K-Nearest Neighbors (KNN)
python training/train_knn.py

# 5. Evaluate all models on held-out test data and generate comparison reports
python training/evaluate.py
```

---

## 11. Empirical Model Comparison Results

Measured on the **identical 161 held-out test images**:

| Model | Architecture / Method | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Status |
|---|---|---|---|---|---|---|
| **Support Vector Machine (SVM)** | RBF Kernel + HOG & HSV Color Features | **81.37%** | **82.47%** | **81.37%** | **81.31%** | Top Performing Model |
| **K-Nearest Neighbors (KNN)** | Euclidean Metric (Optimal $k=3$) | **63.35%** | **68.91%** | **63.35%** | **62.83%** | Distance-based |

> **MCA Research Insight:** On small-scale custom image sets (sub-1000 images), handcrafted domain features (Dalal-Triggs HOG edge textures + HSV color histograms) combined with Support Vector Machines strongly outperform un-pretrained custom CNNs due to the absence of millions of parameters prone to overfitting on limited training instances.

---

## 12. Running the Web Application

Start the Flask server:
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Main Application Views
- **Home Landing Page:** `http://127.0.0.1:5000/`
- **Dashboard:** `http://127.0.0.1:5000/dashboard`
- **Detect Freshness (Upload):** `http://127.0.0.1:5000/upload`
- **Model Comparison:** `http://127.0.0.1:5000/comparison`
- **Prediction History:** `http://127.0.0.1:5000/history`
- **About Project:** `http://127.0.0.1:5000/about`
- **System Health API:** `http://127.0.0.1:5000/health`

---

## 13. Limitations

1. **Surface Inspection Only:** The models evaluate visible exterior optical indicators (skin spots, bruising, discoloration). They cannot detect internal bacterial rot or mold inside intact fruit peel.
2. **Lighting Sensitivity:** Low exposure or harsh direct flash may skew color histogram distributions into the `UNCERTAIN` zone.
3. **Non-Microbiological:** Image classification does not replace microbiological assays or evaluate food safety and bacterial pathogens.

---

## 14. Future Scope

1. **Mobile Edge Deployment:** Quantize the models using TensorFlow Lite for direct on-device execution on Android or Raspberry Pi smart cameras.
2. **Sensor Fusion:** Combine optical images with gas/VOC electronic nose sensors to detect early ripening gases (ethylene) before surface defects appear.
3. **Hyperspectral Imaging:** Utilize Near-Infrared (NIR) imaging to detect internal sugar concentration and core watercore rot.

---

## 15. Student & Project Details

- **Project Title:** FreshGuard AI – Intelligent Food Freshness Detection and Classification System
- **Candidate Name:** Hariom Gupta
- **Candidate UID:** 25MCA20022
- **Degree:** Master of Computer Applications (MCA)
- **Year:** 2026
