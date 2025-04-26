# CreditSaathi

```markdown
# Credit Card Fraud Detection

## 🚀 Objective
The objective of this project is to build an effective machine learning model that can accurately distinguish between legitimate and fraudulent credit card transactions. The goal is to maximize fraud detection accuracy while minimizing false positives.

---

## 📂 Dataset
- **fraudTrain.csv**: Training dataset containing transaction records.
- **fraudTest.csv**: Testing dataset to evaluate the model.
- The dataset includes fields like transaction timestamp, amount, merchant info, user ID, and others.

---

## 🛠️ Steps to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/CreditCard-Fraud-Detection.git
   cd CreditCard-Fraud-Detection
   ```

2. **Install Dependencies**
   Install all required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install manually: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost)*

3. **Run the Notebook**
   - Open the `CreditSaathi.ipynb` Jupyter Notebook.
   - Execute each cell step-by-step.
   - Analyze the visualizations, model performance, and final outcomes.

---

## 📊 Key Features and Workflow

- **Data Preprocessing**:
  - Dropping irrelevant columns.
  - Handling categorical variables using Label Encoding.
  - Scaling features for better model performance.
  
- **Exploratory Data Analysis (EDA)**:
  - Correlation heatmap between features.
  - Visualization of transaction amounts for fraud vs. non-fraud cases.
  - Count plot of fraudulent vs. non-fraudulent transactions.

- **Model Building**:
  - Random Forest Classifier (primary model)
  - XGBoost Classifier (optional secondary model)

- **Model Evaluation**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Curve
  - Precision-Recall Curve
  - Feature Importance plot
  - Misclassification Analysis

---

## ✨ Results

- Built a Random Forest-based fraud detection system.
- Achieved high accuracy and good recall for detecting fraud cases.
- Visualized important features that contribute to fraud detection.
- Identified and analyzed misclassified transactions to understand limitations.

---

## 📌 Requirements

- Python 3.8 or higher
- Libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - xgboost

---

## 📈 Example Visualizations

- 📊 Feature Correlation Heatmap
- 📊 Transaction Amount Distributions
- 📊 Confusion Matrix Visualization
- 📈 ROC Curve
- 📈 Precision-Recall Curve
- 📋 Feature Importance Bar Plot

---

## 🤝 Contributions
This project is developed individually as part of the assigned task for internship assessment.  
For any queries or suggestions, feel free to raise an issue.
Anurag Raut
anuragtraut2003@gmail.com

---

## 📬 Contact
- GitHub Profile: [Your GitHub Username](https://github.com/AnuragRaut08)

```

---
  
# 📦 Ready Project Folder

Here’s what you should have inside your repo:

```
CreditCard-Fraud-Detection/
├── CreditSaathi.ipynb          # Final Jupyter notebook
├── README.md         
├── fraudTrain.csv    # Dataset
├── fraudTest.csv     # Dataset
└── requirements.txt  # (optional, but good practice)
```

➡️ **requirements.txt** content:
```text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```



```markdown
# Credit Card Fraud Detection

## Project Summary
In this project, I built a machine learning model to detect fraudulent credit card transactions using a real-world dataset.  
After careful data preprocessing, exploratory data analysis (EDA), and feature engineering, I trained Random Forest and XGBoost classifiers.  
The model achieved high performance in fraud detection while minimizing false positives.  
Detailed evaluation metrics, visualizations, and misclassification analysis are provided to support model explainability.
```

