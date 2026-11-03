
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "default payment next month"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM": "svm.pkl"
}

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 750;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #AAB4C3;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #E5E7EB;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background-color: #1B2330;
        border: 1px solid #303B4B;
        padding: 17px 15px;
        border-radius: 12px;
        min-height: 105px;
    }

    .metric-label {
        color: #AAB4C3;
        font-size: 0.83rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.65rem;
        font-weight: 700;
    }

    .info-card {
        background-color: #182B35;
        border-left: 4px solid #35B7A2;
        padding: 13px 16px;
        border-radius: 7px;
        color: #DCE8EF;
        margin: 10px 0 16px 0;
    }

    .footer {
        color: #8491A2;
        font-size: 0.8rem;
        text-align: center;
        padding: 25px 0 10px 0;
    }

    div[data-testid="stMetric"] {
        background-color: #1B2330;
        border: 1px solid #303B4B;
        padding: 12px 14px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #AAB4C3;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def get_expected_features(model):
    try:
        return list(
            model.named_steps["preprocessor"].feature_names_in_
        )
    except (AttributeError, KeyError):
        raise ValueError(
            "Unable to identify the expected input features "
            "from the saved model."
        )


def prepare_uploaded_data(df, expected_features):
    df = df.copy()

    df.drop(
        columns=["ID", "AGE_GROUP"],
        errors="ignore",
        inplace=True
    )

    missing_features = [
        col for col in expected_features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing required features: "
            + ", ".join(missing_features)
        )

    X = df[expected_features].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    if X.isnull().any().any():
        invalid_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing or non-numeric values found in: "
            + ", ".join(invalid_columns)
        )

    return X


def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    if len(np.unique(y_true)) == 2:
        metrics["AUC"] = roc_auc_score(
            y_true, y_prob
        )
    else:
        metrics["AUC"] = None

    return metrics


def show_metric_card(label, value):
    display_value = (
        "N/A" if value is None
        else f"{value:.4f}"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(4.5, 3.5),
        dpi=110
    )

    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    cell_colors = [
        ["#16856B", "#D18B28"],
        ["#C74747", "#3478B8"]
    ]

    for i in range(2):
        for j in range(2):
            rectangle = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=cell_colors[i][j],
                edgecolor="#0E1117",
                linewidth=3
            )

            ax.add_patch(rectangle)

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white"
            )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["No Default", "Default"],
        color="#D1D5DB",
        fontsize=9
    )

    ax.set_yticklabels(
        ["No Default", "Default"],
        color="#D1D5DB",
        fontsize=9
    )

    ax.set_xlabel(
        "Predicted Class",
        color="#D1D5DB",
        fontsize=10,
        labelpad=10
    )

    ax.set_ylabel(
        "Actual Class",
        color="#D1D5DB",
        fontsize=10,
        labelpad=10
    )

    ax.set_title(
        "Confusion Matrix",
        color="#F1F5F9",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    return fig, cm


st.markdown(
    '<div class="main-title">Credit Card Default Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        An ML-based application to predict whether a credit card
        customer is likely to default on their next payment.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

st.markdown(
    '<div class="section-title">Upload Test Dataset</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Upload a CSV file containing customer details. "
    "For evaluation metrics, include the actual target column."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=(
        "Upload the held-out test dataset. "
        "The target column is optional for prediction, "
        "but required for model evaluation."
    )
)

st.markdown(
    '<div class="section-title">Select Classification Model</div>',
    unsafe_allow_html=True
)

model_col, description_col = st.columns([1, 2])

with model_col:
    selected_model = st.selectbox(
        "Choose a model",
        list(MODEL_FILES.keys()),
        key="selected_model"
    )

with description_col:
    st.markdown(
        f"""
        <div class="info-card">
            <b>Selected Model:</b> {selected_model}<br>
            Choose one of the five trained classification
            models to generate customer predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded_file is None:
    st.markdown(
        """
        <div class="info-card">
            <b>Getting started</b><br>
            Upload your test CSV and select a classification
            model to generate predictions.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)

    if uploaded_df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    st.success(
        f"Dataset uploaded successfully: "
        f"{len(uploaded_df):,} records"
    )

    with st.expander(
        "Preview Uploaded Dataset",
        expanded=False
    ):
        st.write(
            f"**Dataset shape:** "
            f"{uploaded_df.shape[0]:,} rows × "
            f"{uploaded_df.shape[1]} columns"
        )

        st.dataframe(
            uploaded_df.head(10),
            use_container_width=True
        )

    has_target = TARGET_COLUMN in uploaded_df.columns

    if has_target:
        y_true = pd.to_numeric(
            uploaded_df[TARGET_COLUMN],
            errors="coerce"
        )

        if y_true.isnull().any():
            st.error(
                "The target column contains missing or "
                "non-numeric values."
            )
            st.stop()

        if not set(y_true.unique()).issubset({0, 1}):
            st.error(
                "The target column must contain only 0 and 1."
            )
            st.stop()

        y_true = y_true.astype(int).to_numpy()

        st.success(
            "Actual target labels detected. "
            "Generate predictions to view the evaluation metrics."
        )

    else:
        y_true = None

        st.warning(
            "No actual target labels detected. "
            "Predictions are available, but evaluation "
            "metrics cannot be calculated."
        )

    model_path = MODEL_DIR / MODEL_FILES[selected_model]

    if not model_path.exists():
        st.error(
            f"Saved model not found: {MODEL_FILES[selected_model]}"
        )
        st.stop()

    model = load_model(str(model_path))

    expected_features = get_expected_features(model)

    X = prepare_uploaded_data(
        uploaded_df.drop(
            columns=[TARGET_COLUMN],
            errors="ignore"
        ),
        expected_features
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Generate Predictions",
        type="primary",
        use_container_width=True
    ):
        with st.spinner(
            f"Generating predictions using {selected_model}..."
        ):
            y_pred = model.predict(X)

            probabilities = model.predict_proba(X)

            classes = list(model.classes_)

            if 1 not in classes:
                st.error(
                    "The saved model does not contain "
                    "the expected positive class (1)."
                )
                st.stop()

            positive_class_index = classes.index(1)

            default_probabilities = probabilities[
                :, positive_class_index
            ]

        total_customers = len(y_pred)

        predicted_defaults = int(
            np.sum(y_pred == 1)
        )

        predicted_non_defaults = int(
            np.sum(y_pred == 0)
        )

        st.markdown(
            '<div class="section-title">Prediction Summary</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            show_metric_card(
                "Total Customers",
                total_customers
            )

        with col2:
            show_metric_card(
                "Predicted Defaults",
                predicted_defaults
            )

        with col3:
            show_metric_card(
                "Predicted Non-Defaults",
                predicted_non_defaults
            )

        st.markdown("#### Prediction Distribution")

        distribution_df = pd.DataFrame({
            "Prediction": [
                "No Default",
                "Default"
            ],
            "Customers": [
                predicted_non_defaults,
                predicted_defaults
            ]
        }).set_index("Prediction")

        st.bar_chart(
            distribution_df,
            horizontal=True,
            color="#35B7A2"
        )

        st.markdown(
            '<div class="section-title">Customer Prediction Results</div>',
            unsafe_allow_html=True
        )

        results_df = uploaded_df.drop(
            columns=[TARGET_COLUMN],
            errors="ignore"
        ).copy()

        results_df["Predicted Class"] = y_pred

        results_df["Prediction"] = np.where(
            y_pred == 1,
            "Default",
            "No Default"
        )

        results_df["Default Probability"] = (
            default_probabilities
        )

        if y_true is not None:
            results_df["Actual Class"] = y_true

            results_df["Actual Outcome"] = np.where(
                y_true == 1,
                "Default",
                "No Default"
            )

        st.dataframe(
            results_df.head(20),
            use_container_width=True,
            hide_index=True
        )

        csv_data = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="credit_default_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        if y_true is not None:
            st.divider()

            st.markdown(
                '<div class="section-title">Model Evaluation</div>',
                unsafe_allow_html=True
            )

            metrics = calculate_metrics(
                y_true,
                y_pred,
                default_probabilities
            )

            st.caption(
                f"Model: {selected_model} | "
                "Standard classification threshold: 0.50"
            )

            metric_columns = st.columns(3)

            metric_names = [
                "Accuracy",
                "Precision",
                "Recall",
                "AUC",
                "F1 Score",
                "MCC"
            ]

            for i, metric_name in enumerate(metric_names):
                with metric_columns[i % 3]:
                    show_metric_card(
                        metric_name,
                        metrics[metric_name]
                    )

            st.caption(
                "AUC is calculated using predicted probabilities. "
                "The remaining metrics use the model's predicted classes."
            )

            st.divider()

            st.markdown(
                '<div class="section-title">Confusion Matrix</div>',
                unsafe_allow_html=True
            )

            cm_col1, cm_col2, cm_col3 = st.columns(
                [1, 1.5, 1]
            )

            with cm_col2:
                fig, cm = plot_confusion_matrix(
                    y_true,
                    y_pred
                )

                st.pyplot(
                    fig,
                    use_container_width=False
                )

                plt.close(fig)

            tn, fp, fn, tp = cm.ravel()

            with st.expander(
                "Understand the Confusion Matrix"
            ):
                st.markdown(
                    f"""
                    - **True Negatives:** {tn:,} — Correctly predicted No Default.
                    - **False Positives:** {fp:,} — Predicted Default, but actual outcome was No Default.
                    - **False Negatives:** {fn:,} — Actual Default predicted as No Default.
                    - **True Positives:** {tp:,} — Correctly predicted Default.
                    """
                )

            st.divider()

            st.markdown(
                '<div class="section-title">Classification Report</div>',
                unsafe_allow_html=True
            )

            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "No Default",
                    "Default"
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).T

            st.dataframe(
                report_df.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                }),
                use_container_width=True
            )

        else:
            st.divider()

            st.info(
                "Predictions have been generated successfully. "
                "To calculate Accuracy, AUC, Precision, Recall, "
                "F1-score, MCC, confusion matrix and classification "
                "report, upload a labelled test CSV containing "
                f"the '{TARGET_COLUMN}' column."
            )

except Exception as e:
    st.error(
        f"An error occurred while processing the dataset: {e}"
    )

    st.info(
        "Please verify that the CSV contains all required "
        "features and that the saved model files are available."
    )

st.markdown(
    """
    <div class="footer">
        Credit Card Default Prediction | Machine Learning Classification Project
    </div>
    """,
    unsafe_allow_html=True
)