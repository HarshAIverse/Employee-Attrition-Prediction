<div align="center">

# 🛡️ Palo Alto Networks: Employee Attrition AI Engine

An enterprise-grade, Full-Stack Machine Learning System designed to proactively predict employee turnover, generate detailed risk profiles, and empower HR teams with data-driven retention capabilities.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-1785F8.svg?style=for-the-badge)](https://xgboost.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-Dynamic_Charts-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

</div>

---

## 📖 Overview

Employee attrition incurs massive costs regarding onboarding, training, and lost institutional knowledge. This repository hosts a **Predictive HR Analytics Solution** that shifts talent management strategies from reactive interventions to proactive retention.

By mapping over 30 unique employee data points (such as commute distance, promotion delays, and overtime frequencies), the internal AI models assign individual, real-time "Attrition Risk Scores" from 0 to 100, ensuring HR Business Partners know exactly who to talk to, when to talk to them, and what the core issues are.

---

## ✨ Core Capabilities

- **Intelligent Feature Engineering**: Synthesizes smart metrics natively such as the *Income-to-Experience Ratio*, *Manager Relationship Scores*, and *Work Stress Engines* using raw HR tables.
- **Explainable AI (XAI)**: We don't just supply a probability—the engine dissects the model natively, providing specific "Reason Codes" (e.g., `🚩 High contextual stress: Consistent Overtime triggered.`) utilizing advanced coefficients.
- **Interactive Retention Simulator**: An interactive "What-If" sandbox where HR can virtually adjust parameters (e.g. issuing a 15% salary hike) to see the exact shift in forecasted risk probability.
- **Premium Glassmorphism Dashboard**: A state-of-the-art Streamlit UI featuring deeply themed dark-mode styling, immersive animated dials, and interactive `plotly_dark` density maps.

---

## 🏗️ Architecture & Project Structure

```text
d:\Internship\Employee Attrition Prediction\
├── .streamlit/                 # UI configurations
│   └── config.toml             # Forces native Premium Dark Mode theme
├── data_processing.py          # Object-Oriented ETL and Feature Synthesizer
├── train_model.py              # ML cross-validation suite & pipeline exporter
├── utils.py                    # Streamlit CSS UI injection & scoring helpers
├── app.py                      # Main Streamlit executable dashboard point
├── requirements.txt            # Dependency mappings
├── processed_data.csv          # [Generated] Cleaned target analytics data
├── best_model.pkl              # [Generated] Serialized prediction model
├── feature_meta.pkl            # [Generated] Categorical mappings
└── Documents/                  # Delivery documentation
    ├── research_report.md      # EDA Insights 
    ├── ppt_presentation.md     # Executive meeting slide outline
    └── deployment_guide.md     # Production rollout steps (Docker/AWS)
```

---

## 🧠 Machine Learning Engine

The modeling suite leverages `imblearn.pipeline` with **Synthethic Minority Over-sampling Technique (SMOTE)** to safely resolve target class imbalances (Retained vs. Resigned).

The engine automatically trains and benchmarks:
1. Logistic Regression *(Selected dynamically for maximizing raw Recall)*
2. Decision Trees
3. Random Forest
4. Gradient Boosting
5. XGBoost Classifier

The winning architecture is serialized using `joblib` into a ready-to-serve inference container cleanly integrated into the Streamlit dashboard loop.

---

## 🚀 Installation & Local Execution

Follow these steps to deploy the engine locally:

### 1. Setup Virtual Environment
It is recommended to run this repository inside an isolated virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
Download target backend packages including XGBoost, Imbalanced-learn, and Plotly.
```bash
pip install -r requirements.txt
```

### 3. Generate ML Models & Data Artifacts
Run the modeling backend to clean the raw data and generate `.pkl` files.
```bash
python train_model.py
```

### 4. Boot up the HR Dashboard
Initiate the Streamlit interactive UI.
*(Note: To see the full custom Dark Protocol aesthetics, ensure you use this command from a completely clean terminal session so the `.streamlit/config.toml` overrides apply correctly!)*
```bash
streamlit run app.py
```

---

## 📊 Dashboard Modules 

1. **📈 Executive Dashboard**: Total Firm Risk distributions, Pie charts, and automated aggregate Attrition Rates.
2. **👤 Employee Risk Profile**: Examine individual ID traces via customized SVG-quality risk dials and algorithmic reason logs.
3. **🏢 Department Analytics**: Risk mappings isolated by Job Role and cross-departmental boxplots exposing systematic team stress outliers.
4. **🎛️ Scenario Simulator**: HR adjustment engine allowing simulation of promotions/satisfaction to drop Attrition Probability.
5. **🧠 Explainable AI**: Global absolute predictive weight graphs showing the neural decision hierarchy.

---

<div align="center">
<i>Built for enterprise-grade analytics by Senior AI Developers.</i>
</div>
