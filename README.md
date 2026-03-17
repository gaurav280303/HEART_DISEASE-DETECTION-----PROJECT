<!-- ===========================
     HEART DISEASE PREDICTION - ADVANCED README
     Repo: gaurav-singh-tech/HEART_DISEASE-DETECTION-----PROJECT
     =========================== -->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ef4444,50:111827,100:22c55e&height=230&section=header&text=Heart%20Disease%20Prediction&fontSize=48&fontColor=ffffff&animation=twinkling&fontAlignY=36&desc=Logistic%20Regression%20%7C%20Clinical%20Risk%20Signals%20%7C%20Streamlit%20App%20Deployment&descAlignY=60" />

<p align="center">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-22C55E?style=flat-square" />
  <img src="https://img.shields.io/badge/ML%20Stack-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Artifacts-.pkl%20Model%20%2B%20Scaler-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/Data-918%20Rows-6366F1?style=flat-square" />
</p>

<p>
  <a href="https://heartdisease-detection-madeby-gaurav-singh-bisht.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20App-Open%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="#-how-it-works">
    <img src="https://img.shields.io/badge/Pipeline-See%20How%20It%20Works-0A66C2?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
  <a href="#-how-to-run-locally">
    <img src="https://img.shields.io/badge/Run%20Locally-Setup-22C55E?style=for-the-badge&logo=python&logoColor=white" />
  </a>
</p>

<p>
  <b>Author:</b> Gaurav Singh Bisht
  <br/>
  <sub>“Turning data into life‑saving decisions — one prediction at a time.”</sub>
</p>

<p>
  <img src="https://komarev.com/ghpvc/?username=gaurav-singh-tech&label=Visitors&color=ef4444&style=flat" />
</p>

</div>

---

## ✨ Project Overview
This project delivers a **heart disease risk prediction app** that transforms clinical inputs (age, cholesterol, ECG, exercise angina, etc.) into a clear prediction:

- ✅ **Heart Disease Detected**  
- ✅ **No Heart Disease Detected**

It includes:
- A complete ML workflow (EDA → preprocessing → training → evaluation)
- Model & preprocessing persistence (`.pkl`)
- A production‑style Streamlit UI designed for human-friendly inputs

---

## 🚀 Live App (original link kept exactly)
👉 https://heartdisease-detection-madeby-gaurav-singh-bisht.streamlit.app/

---

## 🧭 Table of Contents
- [Why this project](#-why-this-project)
- [Dataset](#-dataset)
- [How it works](#-how-it-works)
- [Architecture](#-architecture-high-level)
- [Tech stack](#-tech-stack)
- [Artifacts](#-artifacts)
- [Run locally](#-how-to-run-locally)
- [Limitations](#-limitations--disclaimer)
- [Contact](#-contact)

---

## 🎯 Why This Project
Heart disease prediction is a classic applied ML problem — the real challenge is not only training a model, but delivering it as a **usable tool**.

This project emphasizes:
- End‑to‑end ML delivery (not “just a notebook”)
- Clean UX (sliders, dropdowns) instead of confusing raw numeric encodings
- Robustness (handling expected model columns and consistent scaling)

---

## 📊 Dataset
From the included dataset file: `heart (1).csv`

**Shape:** ~918 rows × 12 columns  
Key fields used:
- Age, Sex, ChestPainType
- RestingBP, Cholesterol, FastingBS
- RestingECG, MaxHR, ExerciseAngina
- Oldpeak, ST_Slope  
Target:
- `HeartDisease` (0/1)

---

## ⚙️ How It Works
### ML + App Inference Pipeline
1. User provides health indicators via Streamlit UI
2. Input is assembled into a DataFrame
3. Missing expected columns are safely filled (for stability)
4. Data is aligned to the training feature order
5. Scaler transforms features
6. Logistic Regression predicts risk class

---

## 🏗️ Architecture (High Level)

```mermaid
flowchart LR
  A[User Inputs in Streamlit] --> B[Build Input DataFrame]
  B --> C[Align Columns to Expected Schema]
  C --> D[Scale with Saved Scaler]
  D --> E[Logistic Regression Model]
  E --> F[Risk Prediction Output]
```

---

## 🧰 Tech Stack
<div align="center">

<img src="https://img.shields.io/badge/Python-111827?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-111827?style=for-the-badge&logo=pandas&logoColor=150458" />
<img src="https://img.shields.io/badge/scikit--learn-111827?style=for-the-badge&logo=scikitlearn&logoColor=F7931E" />
<img src="https://img.shields.io/badge/Streamlit-111827?style=for-the-badge&logo=streamlit&logoColor=FF4B4B" />
<img src="https://img.shields.io/badge/Joblib-111827?style=for-the-badge&logo=python&logoColor=white" />

</div>

---

## 📦 Artifacts
| File | Role |
|------|------|
| `logistic_heart_disease_model.pkl` | Trained classifier |
| `scaler.pkl` | Feature scaler used during training |
| `logistic_columns_heart.pkl` | Expected feature schema (column order) |

These artifacts ensure **reproducible predictions** and prevent training/inference mismatch.

---

## 📁 Repository Structure
| File | Purpose |
|------|---------|
| `ML_PROJECT_2,_HEART.ipynb` | EDA + preprocessing + model training |
| `app.py` | Streamlit web app |
| `heart (1).csv` | Dataset |
| `logistic_heart_disease_model.pkl` | Saved model |
| `scaler.pkl` | Saved scaler |
| `logistic_columns_heart.pkl` | Expected feature columns |
| `requirements.txt` | Dependencies |

---

## 🧪 How to Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🧠 Mini Mindmap (1‑minute Recruiter Scan)
```text
Heart Disease Prediction
├── Data
│   └── 918 rows, clinical features + target label
├── Prep
│   ├── Encoding categorical variables
│   ├── Scaling numeric features
│   └── Persisting schema + scaler
├── Model
│   └── Logistic Regression (interpretable baseline)
├── Deployment
│   └── Streamlit UI for real-time predictions
└── Reliability
    └── Column alignment to avoid inference mismatch
```

---

## ⚠️ Limitations & Disclaimer
- This is a **learning + demonstration project** and not a medical device.
- Predictions should **not** be used as medical advice.
- Real clinical deployment would require:
  - extensive validation, bias checks, calibration
  - clinical review and regulatory compliance

---

## 📈 Optional Dynamic Widgets (Developer Branding)
<div align="center">
  <img height="160" src="https://github-readme-stats.vercel.app/api?username=gaurav-singh-tech&show_icons=true&theme=tokyonight" />
  <img height="160" src="https://github-readme-stats.vercel.app/api/top-langs/?username=gaurav-singh-tech&layout=compact&theme=tokyonight" />
</div>

---

## 🤝 Contact
Replace placeholders:
- **GitHub:** https://github.com/gaurav-singh-tech  
- **LinkedIn:** https://www.linkedin.com/in/<your-link>/  
- **Email:** <your-email>  

<div align="center">

### ⭐ If you like this project, consider starring the repo!

</div>
