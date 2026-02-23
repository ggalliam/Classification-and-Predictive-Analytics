# Classification-and-Predictive-Analytics
Classification and Predictive Analytics on the Ontario interest rates

# Ontario Quarterly Tax Interest Rate Direction Classification 

## Project Overview 

This project builds a time-series classification model to predict the quarterly direction (Increase, Decrease, Stable) of the Ontario Underpayment Tax Interest Rate. 

The rate is set quarterly by the Ontario Ministry of Finance. 

--- 
## Target Definition 

Let: 
r_t = Underpayment Interest Rate at quarter t 
Δr_t = r_t - r_{t-1} 
Classification rule: 
Increase if Δr_t > 0 
Decrease if Δr_t < 0 
Stable if Δr_t = 0 
Unit of analysis: One quarter. 
Historical span: 2000Q1–2026Q1. 

--- 
## Research Questions 

RQ1: What is the out-of-sample Macro-F1 score using lagged macroeconomic indicators and historical rate features? 
RQ2: Does adding lagged macroeconomic indicators improve Macro-F1 relative to a historical-rate-only model? 
RQ3: Is model performance stable across economic regimes (2000–2008, 2009–2019, 2020–2026)? 
RQ4: Which predictors most influence classification decisions? 

--- 
## Data Sources 
Core dataset: 
Ontario Data Catalogue – Tax Interest Rates 
External indicators: 
- Statistics Canada (CPI, GDP, unemployment) 
- Bank of Canada (policy rate)
---
## Evaluation Strategy

Time-based walk-forward validation: 
Train: 2000Q1–2015Q4 → Test: 2016Q1 
Rolling forward one quarter at a time. 
No random splitting. 

--- 
## Baselines 
1. Always predict "Stable"
2. Predict same direction as previous quarter
---
## Metrics 

Primary metric: Macro-F1 score 
Secondary: Accuracy, Balanced Accuracy, Confusion Matrix, ROC-AUC 

--- 
## Validation Strategy

To avoid leakage, time-based walk-forward validation is used.
Baseline models:
1) No change, always predicted as "Stable"
2) Last-quarter direction
Primary Evaluation Metric: Macro-F1 score
---
## Folder Structure 
data/ 
notebooks/ 
src/ 

Raw dataset:
[Ontario_tax_rates.csv](https://github.com/user-attachments/files/25500994/Ontario_tax_rates.csv)

Processed dataset:
[01_data_cleaning_and_eda.ipynb](https://github.com/user-attachments/files/25473630/01_data_cleaning_and_eda.ipynb)

---
## Tools 

Python, P[01_data_cleaning_and_eda.ipynb]
andas, NumPy, Scikit-learn, SHAP, Matplotlib


