# CodeAlpha Task 4: Disease Prediction from Medical Data

A machine learning project that predicts whether a tumor is **malignant or
benign** from patient measurements, built as part of the CodeAlpha Machine
Learning internship.

## Dataset
- Breast Cancer Wisconsin dataset (UCI ML Repository, loaded via scikit-learn)
- 569 patients, 30 numeric features (radius, texture, perimeter, area, etc.)
- Target: 0 = Malignant, 1 = Benign
- No missing values

## Approach
1. Explored the data (class balance, statistics, correlation heatmap)
2. Split into 80% training and 20% testing (stratified)
3. Scaled features with StandardScaler
4. Trained three models: Logistic Regression, SVM and Random Forest
5. Evaluated with Accuracy, Precision, Recall, F1-Score and ROC-AUC
6. Analysed errors with a confusion matrix and feature importance

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| SVM | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| Random Forest | 0.956 | 0.959 | 0.972 | 0.966 | 0.993 |

**Best models:** Logistic Regression and SVM (tied), with 98.2% accuracy.

## Key Findings
- Logistic Regression and SVM tied, and both beat Random Forest, which
  suggests the scaled data is close to linearly separable.
- All models have ROC-AUC above 0.99, so they separate the two classes very
  well.
- The most important features according to Random Forest were
   worst parameters, worst area and worst-concave points.
- In medical prediction, a false negative (a malignant tumor predicted as
  benign) is the most serious error, so the confusion matrix was used to
  check this.

## Conclusion
Machine learning can predict this disease with high accuracy from
measurements alone, and simple models can match complex ones on clean,
structured data. Such a model should support doctors, not replace them.
Future improvements: cross-validation, hyperparameter tuning and testing on
a larger dataset.

## Tools Used
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab

## How to Run
1. Open `CodeAlpha_DiseasePrediction.ipynb` in Google Colab
2. Click **Runtime → Run all**

## Author
Abishek SM
Machine Learning Intern at CodeAlpha
