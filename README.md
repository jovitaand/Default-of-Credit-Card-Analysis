# Analysis of Default of Credit Card Clients

## Objective
The goal of this analysis is to predict the likelihood of credit card default by clients based on their demographic and financial features. Additionally, we assess model fairness and interpretability to ensure transparency and ethical outcomes.

---

## Dataset Overview

### Source
The dataset contains information on 30,000 credit card clients in Taiwan and includes 25 features such as:
- **Demographic attributes**: AGE, SEX, EDUCATION, MARRIAGE
- **Financial attributes**: LIMIT_BAL, BILL_AMT (bill statements), PAY_AMT (payment amounts)
- **Repayment status**: PAY_0 through PAY_6
- **Target variable**: `default.payment.next.month` (1 = default, 0 = no default)

### Key Statistics
- Total records: 30,000
- Class distribution:
  - Default: ~22%
  - Non-default: ~78%

---

## Methodology

### 1. Data Preprocessing
- **Missing Values**: None detected in the dataset.
- **Feature Scaling**: Applied `RobustScaler` to normalize skewed features like LIMIT_BAL and BILL_AMT.
- **Outlier Handling**: Addressed extreme values in features such as CREDIT_UTILIZATION and PAYMENT_TO_BILL_RATIO.
- **Class Balancing**: Used oversampling to balance the class distribution, ensuring equal representation of default and non-default cases.

### 2. Feature Engineering
Enhanced the dataset with the following derived features:
- **CREDIT_UTILIZATION**: Total bill amounts divided by credit limit.
- **AVG_REPAY_DELAY**: Average repayment delay across six months.
- **MAX_REPAY_DELAY**: Maximum repayment delay.
- **TOTAL_PAYMENTS**: Sum of payment amounts across six months.
- **PAYMENT_TO_BILL_RATIO**: Ratio of total payments to total bills.

---

## Models and Evaluation

### Models Used
1. **Random Forest**
2. **Logistic Regression**
3. **XGBoost**
4. **Neural Network**

### Performance Metrics
- **Confusion Matrix**: To evaluate true positives, true negatives, false positives, and false negatives.
- **Balanced Accuracy**: Accounts for class imbalance.
- **Precision, Recall, F1-score**: To assess model effectiveness across both classes.

| Model              | Balanced Accuracy | Precision (Default) | Recall (Default) | F1-score (Default) |
|--------------------|-------------------|---------------------|------------------|--------------------|
| Random Forest      | 66.66%           | 0.60                | 0.41             | 0.49               |
| Logistic Regression| 67.04%           | 0.61                | 0.42             | 0.50               |
| XGBoost            | 68.51%           | 0.48                | 0.54             | 0.51               |
| Neural Network     | TBD              | TBD                 | TBD              | TBD                |

---

## Explainability

### SHAP Analysis
- **Global Importance**:
  - Features like CREDIT_UTILIZATION and TOTAL_BILLS have the highest contribution to predictions.
  - Demographic features (SEX, MARRIAGE) have minimal impact.
- **Local Importance**:
  - For individual predictions, high repayment delays significantly influence default likelihood.

### LIME Explanation
- Provided case-specific interpretations for individual predictions.
- Demonstrated transparency by explaining how each feature affected the prediction for a sample client.

---

## Fairness Analysis
### Metrics
- **Demographic Parity Difference**: Measures the difference in positive prediction rates across gender groups.
- **Equalized Odds Difference**: Evaluates discrepancies in error rates (false positives and negatives) across groups.

### Findings
- **Demographic Parity Difference**: 0.12
- **Equalized Odds Difference**: 0.09
- Some bias was detected, suggesting room for improvement in ensuring fairness.

---

## Conclusion
- **Best Model**: XGBoost showed the highest balanced accuracy and interpretability.
- **Fairness**: While the models performed well overall, slight biases exist that need mitigation.
- **Recommendations**:
  1. Apply post-processing fairness techniques.
  2. Use SHAP and LIME insights to refine features and thresholds.
  3. Regularly monitor model performance for fairness and drift.

---

## Future Work
- Explore advanced ensemble techniques.
- Implement real-time monitoring of model fairness.
- Incorporate additional features to enhance predictive power.
  
