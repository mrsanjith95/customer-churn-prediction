# 📊 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-red?logo=streamlit)

A machine learning application that predicts whether a bank customer is likely to churn based on demographic and banking information. The project features exploratory data analysis, evaluation of multiple ML models, model persistence, and an interactive Streamlit web dashboard.

---

## 🎓 CodSoft Internship — Task 3

This project was developed as part of the **CodSoft Machine Learning Internship** (**Task 3: Customer Churn Prediction**).

---

## ✨ Key Highlights

- **10,000 Customer Records**: Analyzed historical banking dataset (`Churn_Modelling.csv`).
- **10 Input Features**: Utilized key demographic and financial attributes for modeling.
- **3 ML Algorithms Evaluated**: Compared Logistic Regression, Random Forest, and Gradient Boosting.
- **Gradient Boosting Model**: Selected as the top-performing model on the held-out test split.
- **Streamlit Interactive Dashboard**: Compact, single-screen user interface for real-time predictions.

---

## 🔗 Project Links

- **GitHub Repository**: [https://github.com/mrsanjith95/customer-churn-prediction](https://github.com/mrsanjith95/customer-churn-prediction)

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.11+
- **Data Analysis & Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Joblib
- **Web Application Framework**: Streamlit
- **Environment & Tools**: Jupyter Notebook
- **Version Control & Hosting**: Git, GitHub

---

## 📌 Project Overview

Customer churn occurs when customers stop doing business with a service provider. In the banking sector, identifying customers at risk of leaving allows financial institutions to take proactive retention steps. 

This project aims to:
1. Conduct Exploratory Data Analysis (EDA) to discover key risk indicators for customer churn.
2. Build data preprocessing pipelines (StandardScaler for numerical, OneHotEncoder for categorical variables).
3. Train and compare multiple machine learning algorithms.
4. Deploy the best-performing model into an intuitive, real-time dashboard application.

---

## 📊 Dataset & Features

The model is trained on the `Churn_Modelling.csv` dataset containing 10,000 customer entries.

### Target Variable
- **`Exited`**: `0` = Stayed, `1` = Churned

### 10 Input Features Used
1. **`CreditScore`**: Customer's credit score (300–850).
2. **`Geography`**: Customer location (`France`, `Germany`, `Spain`).
3. **`Gender`**: Customer gender (`Female`, `Male`).
4. **`Age`**: Customer age in years (18–100).
5. **`Tenure`**: Number of years the customer has been with the bank (0–10).
6. **`Balance`**: Account balance ($).
7. **`NumOfProducts`**: Number of bank products used (1–4).
8. **`HasCrCard`**: Credit card holder status (`1` = Yes, `0` = No).
9. **`IsActiveMember`**: Active membership status (`1` = Yes, `0` = No).
10. **`EstimatedSalary`**: Estimated annual salary ($).

*Note: Non-predictive identifiers (`RowNumber`, `CustomerId`, `Surname`) and temporary EDA variables (`AgeGroup`) were dropped prior to model training.*

---

## 🔍 Exploratory Data Analysis (EDA) Findings

Key insights established from the dataset analysis:
- **Overall Churn Rate**: 79.63% of customers stayed, while 20.37% churned.
- **Geographic Differences**: Customers in **Germany** exhibited a higher observed churn rate compared to those in France and Spain.
- **Gender Insights**: Female customers showed a higher observed churn rate than male customers in this dataset.
- **Member Activity**: Inactive members (`IsActiveMember = 0`) had a higher observed churn rate than active members.
- **Account Balance**: Churned customers had a higher average account balance compared to retained customers.
- **Product Volume**: Customers holding 3 or 4 products showed very high observed churn rates. *(Note: These product tiers contain relatively few customer records and should be interpreted cautiously).*

---

## ⚙️ Data Preprocessing & Modeling Pipeline

1. **Preprocessing Pipeline**:
   - **Numerical Features**: Scaled using `StandardScaler`.
   - **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown="ignore")`.
2. **Train-Test Split**:
   - Split ratio: **80% Train / 20% Test** (`test_size = 0.20`).
   - Stratified sampling (`stratify = y`, `random_state = 42`) to preserve target distribution.

---

## 📈 Model Performance & Evaluation

Three classification models were trained and evaluated on the held-out test dataset (2,000 samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 80.80% | 58.91% | 18.67% | 28.36% | 77.48% |
| **Random Forest** | 85.90% | 76.60% | 44.23% | 56.07% | 85.32% |
| **Gradient Boosting Classifier** | **87.05%** | **79.37%** | **49.14%** | **60.70%** | **86.97%** |

### Selected Final Model
The **Gradient Boosting Classifier** achieved the highest overall scores across Accuracy (87.05%), F1-Score (60.70%), and ROC-AUC (86.97%) on this project's held-out test split, and was serialized into `models/customer_churn_model.pkl`.

### Performance Metrics Explained
- **Accuracy**: Overall proportion of correctly predicted customers (churned and stayed).
- **Precision**: Proportion of predicted churners who actually churned.
- **Recall**: Proportion of actual churners correctly identified by the model.
- **F1-Score**: Harmonic mean of Precision and Recall, balancing false positives and false negatives.
- **ROC-AUC**: Ability of the model to distinguish between churning and non-churning customers across decision thresholds.

---

## 🖥️ Streamlit Web Application

The project includes an interactive dashboard (`app.py`) built with Streamlit.

### Key Application Features
- **Customer Information Input**: Convenient controls for entering all 10 customer features.
- **Real-Time Prediction**: Instant prediction output (`🟢 Likely to Stay` vs. `🔴 Likely to Churn`).
- **Probability Breakdown**: Numerical metrics for Stay vs. Churn probabilities.
- **Probability Visual Comparison**: Horizontal progress bar comparing Stay vs. Churn rates.
- **Risk Level & Insight**: Categorization into Low, Medium, or High risk with text summaries.
- **Model Metadata**: Collapsible section displaying model parameters and test metrics.
- **Compact Dashboard Layout**: Single-viewport design optimized for desktop screens.

---

## 📸 Application Preview

![Customer Churn Prediction Dashboard](screenshots/dashboard.png)

---

## 📁 Project Structure

```text
Customer Churn Prediction/
│
├── dataset/
│   └── Churn_Modelling.csv         # Raw customer dataset (10,000 records)
│
├── models/
│   └── customer_churn_model.pkl    # Serialized Gradient Boosting model pipeline
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb # Data analysis, preprocessing & model training
│
├── screenshots/
│   └── dashboard.png               # Streamlit application UI screenshot
│
├── app.py                           # Streamlit web application
├── requirements.txt                 # Exact package dependency versions
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore configuration
```

---

## 🚀 Installation & Execution

### Prerequisites
- Python 3.11+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/mrsanjith95/customer-churn-prediction.git
cd "Customer Churn Prediction"
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell / Command Prompt)
.\.venv\Scripts\activate

# Activate on macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## ⚙️ Model & Dependency Compatibility

The serialized model (`customer_churn_model.pkl`) relies on exact scikit-learn pipeline specifications:
- **`scikit-learn`**: `1.6.1`
- **`joblib`**: `1.6.0`
- **`pandas`**: `3.0.6`
- **`numpy`**: `2.4.6`
- **`streamlit`**: `1.64.0`

Ensure you install dependencies using `pip install -r requirements.txt` to maintain version compatibility.

---

## ⚠️ Limitations

- **Dataset Scope**: Trained exclusively on the provided historical dataset (`Churn_Modelling.csv`).
- **Probabilistic Estimates**: Predictions reflect statistical probabilities based on historical patterns and should not be treated as absolute certainty.
- **Test Set Dependency**: Metrics are specific to the 20% test split evaluated in this project.
- **Behavioral Evolution**: The model does not dynamically account for external market trends or policy changes outside the training dataset.

---

## 🔮 Future Improvements

- **Hyperparameter Tuning**: Perform systematic grid search / Bayesian optimization to tune decision tree depths and learning rates.
- **Cross-Validation**: Implement k-fold cross-validation during model evaluation.
- **Threshold Optimization**: Adjust classification decision thresholds to optimize Recall for churn detection based on business costs.
- **Feature Engineering**: Incorporate additional behavioral metrics such as transaction frequency and customer service interactions.
- **Model Explainability**: Integrate SHAP (SHapley Additive exPlanations) or LIME for feature importance breakdown per customer.
- **Cloud Deployment**: Deploy the Streamlit app to Streamlit Community Cloud or AWS/GCP.
