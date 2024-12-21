# Analysis of Default of Credit Card Clients

## Objective
The analysis aims to:
1. Predict the likelihood of credit card default based on demographic and financial features.
2. Assess model fairness and interpretability to ensure ethical and transparent outcomes.

---

## Dataset Overview
- **Source:** Data on 30,000 credit card clients in Taiwan.
- **Features:** 
  - **Demographic Attributes:** AGE, SEX, EDUCATION, MARRIAGE
  - **Financial Attributes:** LIMIT_BAL, BILL_AMT, PAY_AMT
  - **Repayment Status:** PAY_0 to PAY_6
  - **Target Variable:** `default.payment.next.month` (1 = Default, 0 = No Default)
- **Key Statistics:**
  - **Total Records:** 30,000
  - **Class Distribution:** 
    - Default: ~22%
    - Non-default: ~78%

---

## Methodology

### 1. Data Preprocessing
- **Missing Values:** None detected.
- **Feature Scaling:** Applied `RobustScaler` to normalize skewed features.
- **Outlier Handling:** Addressed extreme values in CREDIT_UTILIZATION and PAYMENT_TO_BILL_RATIO.
- **Class Balancing:** Oversampled minority class to address imbalance.

### 2. Feature Engineering
Derived features added:
- **CREDIT_UTILIZATION:** Total bill amounts divided by credit limit.
- **AVG_REPAY_DELAY:** Average repayment delay across six months.
- **MAX_REPAY_DELAY:** Maximum repayment delay.
- **TOTAL_PAYMENTS:** Total payment amounts across six months.
- **PAYMENT_TO_BILL_RATIO:** Ratio of total payments to total bills.

---

## Models and Evaluation

### Models Used
1. Random Forest
2. Logistic Regression
3. XGBoost
4. Neural Network

### Performance Metrics
- **Confusion Matrix:** True positives, true negatives, false positives, and false negatives.
- **Balanced Accuracy:** Accounts for class imbalance.
- **Precision, Recall, F1-Score:** Assesses effectiveness across both classes.

| **Model**            | **Accuracy** |
|-----------------------|--------------|
| Random Forest         | 81%          |
| Logistic Regression   | 70%          |
| XGBoost               | 77%          |
| Neural Network        | 50%          |

---

## Explainability and Fairness

### Explainability
- **SHAP Analysis:**
  - **Global Importance:** Features like CREDIT_UTILIZATION and TOTAL_BILLS contribute most to predictions.
  - **Local Importance:** High repayment delays influence individual default likelihoods.
- **LIME Explanation:**
  - Provided case-specific interpretations for transparency.

### Fairness Analysis
- **Metrics:**
  - Demographic Parity Difference: 0.12
  - Equalized Odds Difference: 0.09
- **Findings:** 
  - Slight biases detected, with women and older individuals flagged disproportionately as high-risk.

---

## Conclusions
- **Best Model:** XGBoost demonstrated the best balance of accuracy and interpretability.
- **Fairness Issues:** Biases in demographic and financial features require mitigation.
- **Recommendations:**
  1. Apply fairness post-processing techniques.
  2. Use SHAP and LIME insights to refine features and thresholds.
  3. Regularly monitor for fairness and data drift.

---

## Future Work
- Explore advanced ensemble methods.
- Implement real-time fairness monitoring.
- Incorporate additional features to enhance model performance.
