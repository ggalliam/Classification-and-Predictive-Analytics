# Classification-and-Predictive-Analytics
Classification and Predictive Analytics on the Ontario interest rates

# Ontario Quarterly Tax Interest Rate Direction Classification 

## Project Overview 

This project builds a time-series classification model to predict the quarterly direction (Increase, Decrease, Stable) of the Ontario Underpayment Tax Interest Rate. 

The rate is set quarterly by the Ontario Ministry of Finance. 

--- 

--- 
## Research Questions 

- RQ1: What is the out-of-sample Macro-F1 score using lagged macroeconomic indicators and historical rate features? 
- RQ2: Does adding lagged macroeconomic indicators improve Macro-F1 relative to a historical-rate-only model? 
- RQ3: Is model performance stable across economic regimes (2000–2008, 2009–2019, 2020–2026)? 
- RQ4: Which predictors most influence classification decisions? 

--- 
## Data Selection 

Interest tax rates are the main control tool in the macroeconomics of the modern world. They  function as policy instruments to influence macroeconomic stability, regulate inflation, and guide economic growth. Governments shape borrowing plans because of them, companies adjust spending around their shifts, people rethink savings when they move. Every three months in Ontario, the Finance Ministry adjusts tax-related percentages which are the quiet markers of what the economy is facing at the moment (Ontario Ministry of Finance, 2026).


Furthermore, the dataset used in this analysis is sourced from the Ontario Data Catalogue which contains over a hundred data points of quarterly tax interest rates, such as overpayment rates (earned interest), appeal rates, and underpayment rates (interest charged).  This dataset is providing a view of quarterly interest rate behaviour from 1998 to 2026. 


For the first quarter of 2026, the mean interest rate is 7.25%, indicating a moderately high rate environment. Interest rates trend upward, suggesting a gradual tightening of fiscal conditions, potentially reflecting macroeconomic inflation control policies.

<img width="812" height="395" alt="Screenshot 2026-04-26 at 7 41 57 PM" src="https://github.com/user-attachments/assets/8e47052e-e437-4d03-b381-c392eed2ce4c" />

Most rate changes are minor, so the policy adjustments are incremental rather than abrupt. Outliers are minimal and may represent policy interventions or external economic shocks.


Taking a look at the interest tax rates over the years, it shows a bigger picture. The standard deviation is 0.42%, suggesting low short-term volatility. The minimum interest rate is 6.8%, and the maximum is 7.9%. 


Overall, the relatively small changes in rates suggest a stable policy environment with controlled adjustments over time. 


From descriptive statistics, it is known that the average interest rate is about 7.25% which is suggesting a stable policy trend with subtle adjustments and no major changes.

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
docs/
figures/
notebooks/ 
README.md/
requirements.txt/

Raw dataset:
[Ontario_tax_rates.csv](https://github.com/user-attachments/files/25500994/Ontario_tax_rates.csv)

Processed dataset:
[01_data_cleaning_and_eda.ipynb](https://github.com/user-attachments/files/25473630/01_data_cleaning_and_eda.ipynb)

---
## Tools 

Python, P[01_data_cleaning_and_eda.ipynb]
andas, NumPy, Scikit-learn, SHAP, Matplotlib


