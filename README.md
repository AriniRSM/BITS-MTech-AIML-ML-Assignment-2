# Credit Card Default Prediction Using Machine Learning

Binary Classification Project predicting whether a credit card user will default on their payment in the coming month. This project
uses five machine learning models, evaluates their performances using six different metrics, and gives an interactive Streamlit app
to make predictions.
## Project Links

- **GitHub Repository:** https://github.com/AriniRSM/BITS-MTech-AIML-ML-Assignment-2
- **Live Streamlit Application:** https://credit-default-predictor-ml-assignment.streamlit.app/
- **Dataset:** [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## 1. Problem Statement

The objective of the project is to design and compare the performance of supervised learning methods that are used to predict if
there will be default payment by the credit card customer in the next month. It is a binary classification problem where the target
variable “default payment next month” has the following classes: 0 – the customer did not make default in the next month. 1 – the
customer made default in the next month. In this project, the comparison of the five different classifiers is performed using various
evaluation metrics.

## 2. Dataset Description

**Dataset:** Default of Credit Card Clients  
**Source:** UCI Machine Learning Repository  
**Task:** Binary Classification  
**Target variable:** `default payment next month`

The dataset contains 30,000 customer records and 23 input features.

| Attribute | Description |
|---|---|
| Total instances | 30,000 |
| Input features | 23 |
| Target variable | `default payment next month` |
| Class 0 | No Default |
| Class 1 | Default |

### Target Class Distribution

| Class | Description | Instances | Percentage |
|---|---|---:|---:|
| 0 | No Default | 23,364 | 77.88% |
| 1 | Default | 6,636 | 22.12% |

The target classes are imbalanced. Consequently, accuracy alone is not sufficient to understand model performance, and metrics such as precision, recall, F1 Score, AUC, and MCC are also considered.

## 3. Exploratory Data Analysis

Exploratory data analysis is presented in `notebooks/eda.ipynb`. It covers the following: data overview, feature distributions,
distribution of target class, and correlation between customer features and defaults.

Model evaluation and analysis are presented in `notebooks/model_analysis.ipynb`.

## 4. Data Preprocessing

The following preprocessing steps are used:

- **Data inspection:** Examined dataset dimensions, feature types, missing values, duplicate records, and target distribution.
- **Feature removal:** Removed the `ID` column because it is an identifier rather than a predictive customer attribute. `AGE_GROUP`, used for exploratory analysis, is excluded from model training.
- **Categorical encoding:** Applied One-Hot Encoding to `SEX`, `EDUCATION`, and `MARRIAGE`.
- **Numerical scaling:** Applied StandardScaler to numerical features for Logistic Regression, KNN, and SVM.
- **Train-test split:** Used an 80:20 training and testing split.
- **Stratification:** Preserved the target-class proportions in the training and testing sets using stratified sampling.
- **Reproducibility:** Used `random_state=42` for the split.

Preprocessing is integrated with the classifiers using Scikit-learn pipelines. This ensures that the same transformations are applied when the saved models are used for prediction.

## 5. Machine Learning Models

Five classification algorithms were implemented:

| Model | Description |
|---|---|
| Logistic Regression | A linear classification algorithm that estimates the probability of a customer defaulting. |
| Decision Tree | A tree-based classifier that learns decision rules from the training data. |
| K-Nearest Neighbors (KNN) | Predicts the class based on the nearest training samples. |
| Gaussian Naive Bayes | A probabilistic classifier based on Bayes' theorem and the Gaussian feature-distribution assumption. |
| Support Vector Machine (SVM) | A classifier that learns a decision boundary separating the classes. |

The SVM implementation uses a linear Support Vector Classifier with probability calibration through `CalibratedClassifierCV`.

Each model is trained and evaluated using the same stratified train-test split. The trained preprocessing and classification pipelines are saved as Joblib files in the `models/` directory.

## 6. Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Proportion of all observations classified correctly. |
| AUC | Measures the model's ability to distinguish between the two classes across classification thresholds. |
| Precision | Of the customers predicted as defaulters, the proportion who actually defaulted. |
| Recall | Of the actual defaulters, the proportion correctly identified by the model. |
| F1 Score | Harmonic mean of precision and recall. |
| Matthews Correlation Coefficient (MCC) | A balanced measure of binary classification quality that considers all four confusion-matrix outcomes. |

Precision, Recall, and F1 Score in the comparison below refer to the positive class, **Default (Class 1)**.

## 7. Model Performance Comparison

The results below are from the held-out test set containing 6,000 observations. Classification metrics use the models' default class predictions, with a standard classification threshold of 0.50. AUC is calculated using predicted probabilities.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8088 | 0.7100 | 0.6923 | 0.2442 | 0.3610 | 0.3302 |
| Decision Tree | 0.7153 | 0.6016 | 0.3669 | 0.3956 | 0.3807 | 0.1965 |
| KNN | 0.7948 | 0.7040 | 0.5561 | 0.3587 | **0.4361** | 0.3292 |
| Gaussian Naive Bayes | 0.4163 | 0.6516 | 0.2495 | **0.8161** | 0.3821 | 0.1104 |
| SVM | 0.8080 | 0.6927 | 0.6834 | 0.2457 | 0.3614 | 0.3273 |


### Best Result by Metric

| Metric | Best-performing model | Score |
|---|---|---:|
| Accuracy | Logistic Regression | 0.8088 |
| AUC | Logistic Regression | 0.7100 |
| Precision | Logistic Regression | 0.6923 |
| Recall | Gaussian Naive Bayes | 0.8161 |
| F1 Score | KNN | 0.4361 |
| MCC | Logistic Regression | 0.3302 |

## 8. Model-wise Observations

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest Accuracy (0.8088), AUC (0.7100), Precision (0.6923), and MCC (0.3302). However, its Recall (0.2442) for the Default class is relatively low, indicating that it misses many actual defaulters. |
| **Decision Tree** | Achieved an Accuracy of 0.7153 and Recall of 0.3956. Its overall performance was moderate, with an F1 Score of 0.3807 and MCC of 0.1965. |
| **K-Nearest Neighbors (KNN)** | Achieved the highest F1 Score (0.4361) for the Default class, with Precision of 0.5561 and Recall of 0.3587. It provides the best balance between Precision and Recall among the evaluated models, based on F1 Score. |
| **Gaussian Naive Bayes** | Achieved the highest Recall (0.8161) for the Default class. However, its Precision (0.2495) and Accuracy (0.4163) were low, indicating a high number of false-positive predictions. |
| **Support Vector Machine (SVM)** | Achieved an Accuracy of 0.8080 and Precision of 0.6834, performing similarly to Logistic Regression. However, its Default-class Recall (0.2457) was relatively low. |
| **Overall Winner for the Dataset** | No single model performs best across all metrics. KNN achieved the highest Default-class F1 Score (0.4361), while Logistic Regression achieved the highest Accuracy, AUC, Precision, and MCC. KNN is the preferred model when Default-class F1 Score is the selection priority. |

### Logistic Regression

Logistic Regression achieved the highest accuracy, AUC, precision, and MCC among the five models. Its default-class precision was 0.6923, but its recall was 0.2442. This means that although its positive predictions were comparatively reliable, it missed many actual defaulters.

### Decision Tree

The Decision Tree classifier achieved an accuracy of 0.7153 and a default-class recall of 0.3956. Its results were moderate compared with the other classifiers, with a default-class F1 Score of 0.3807.

### K-Nearest Neighbors (KNN)

KNN achieved the highest default-class F1 Score of 0.4361, with precision of 0.5561 and recall of 0.3587. Among the evaluated models, it provided the strongest balance of precision and recall for the default class according to F1 Score.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved the highest default-class recall of 0.8161. However, its precision was 0.2495, indicating that many customers predicted as defaulters were actually non-defaulters. This trade-off is also reflected in its lower accuracy and MCC.

### Support Vector Machine (SVM)

SVM achieved an accuracy of 0.8080 and default-class precision of 0.6834. Its results were close to Logistic Regression, with relatively high precision but low default-class recall of 0.2457.

## 9. Overall Analysis

There was no one model that performed best based on all evaluation metrics.
The logistic regression model performed best in terms of accuracy, AUC, precision, and MCC. But from its low recall score, we know
that it doesn’t find a large number of actual defaulters.

KNN performed best in terms of F1 Score for the default class, meaning it is balanced in terms of precision and recall compared to
others. Gaussian Naive Bayes found the largest number of actual defaulters, but at the same time generated a lot of false positives.
In the case of using the model for credit risk applications, we have to make sure that the model will not ignore a large number of
actual defaulters and classify a non-defaulter as a defaulter. We should look at the trade-off between precision and recall and not
only focus on accuracy.

Considering the above findings, KNN performed best in terms of default-class F1 Score among the examined models, whereas
logistic regression performed best in several evaluation metrics.

## 10. Streamlit Application

The trained models are deployed through an interactive Streamlit application.

**Live Application:** https://credit-default-predictor-ml-assignment.streamlit.app/

### Application Features

- **CSV Upload:** Upload a customer test dataset in CSV format.
- **Model Selection:** Select one of the five trained classification models.
- **Prediction Generation:** Generate predictions using the selected saved model.
- **Prediction Summary:** View the total number of customers predicted as Default and No Default.
- **Prediction Distribution:** View the distribution of predicted classes.
- **Customer-level Results:** Inspect individual customer predictions and default probabilities.
- **Download Results:** Download the prediction results as a CSV file.
- **Model Evaluation:** When the uploaded dataset contains the actual target column, view Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- **Confusion Matrix:** Visualize true negatives, false positives, false negatives, and true positives.
- **Classification Report:** View class-wise precision, recall, F1 Score, and support.

The application loads the saved model pipelines from the `models/` directory. It does not retrain models on uploaded data.

The target column is optional for generating predictions, but it is required to calculate evaluation metrics and display the confusion matrix and classification report.

The repository's `test_data.csv` contains the held-out test features along with their actual target labels. It can be uploaded to the Streamlit application to generate predictions and calculate evaluation metrics, including the confusion matrix and classification report.

## 11. Repository Structure

```text
BITS-MTech-AIML-ML-Assignment-2/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── default of credit card clients.xls
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── create_test_csv.py
│   ├── inspect_dataset.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── logistic_regression_model.py
│   ├── decision_tree_model.py
│   ├── knn_model.py
│   ├── naive_bayes_model.py
│   ├── svm_model.py
│   └── train_models.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── gaussian_naive_bayes.pkl
│   └── svm.pkl
│
├── test_data.csv
└── model_comparison.csv
```

## 12. Installation and Execution

### Prerequisites

- Python
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/AriniRSM/BITS-MTech-AIML-ML-Assignment-2.git
```

### Step 2: Navigate to the project directory

```bash
cd BITS-MTech-AIML-ML-Assignment-2
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser. The saved model files in `models/` must be present for predictions to work.

### Optional: Train the models again

To reproduce the training and evaluation workflow from the project root:

```bash
python -m src.train_models
```

This script loads the dataset, performs the stratified train-test split, trains the five classification pipelines, saves the trained models, generates `test_data.csv`, and writes the model metrics to `model_comparison.csv`.

## 13. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Joblib | Saving and loading trained model pipelines |
| Streamlit | Interactive web application |
| Jupyter Notebook | Exploratory analysis and model analysis |
| Git and GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application deployment |

## 14. Conclusion

This project implements and compares five supervised machine learning classification algorithms for credit card default prediction. It demonstrates the machine learning workflow, including exploratory data analysis, preprocessing, stratified splitting, model training, evaluation, model persistence, and deployment.

The results highlight the importance of evaluating classification models using multiple metrics, particularly when the target classes are imbalanced. Logistic Regression achieved the strongest results in accuracy, AUC, precision, and MCC; KNN achieved the highest F1 Score for the default class; and Gaussian Naive Bayes achieved the highest default-class recall.

The Streamlit application makes the trained models accessible through a simple interface for uploading customer datasets, generating predictions, and evaluating results when actual labels are available.