
import streamlit as st
import pandas as pd
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


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Default Prediction")

st.write(
    "An ML-based application to predict whether a "
    "credit card customer is likely to default "
    "on their next payment."
)

st.divider()


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing customer details",
    type=["csv"]
)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.header("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys())
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset uploaded successfully: "
            f"{len(uploaded_df)} records"
        )

        # Display uploaded dataset
        with st.expander("Preview uploaded dataset"):
            st.dataframe(uploaded_df.head(10))

        # ------------------------------------------
        # SEPARATE ACTUAL TARGET LABELS
        # ------------------------------------------

        y_true = None

        if TARGET_COLUMN in uploaded_df.columns:

            y_true = uploaded_df[TARGET_COLUMN].copy()

            uploaded_df = uploaded_df.drop(
                columns=[TARGET_COLUMN]
            )

        # ------------------------------------------
        # REMOVE ID COLUMN
        # ------------------------------------------

        if "ID" in uploaded_df.columns:

            uploaded_df = uploaded_df.drop(
                columns=["ID"]
            )

        # ------------------------------------------
        # LOAD SELECTED MODEL
        # ------------------------------------------

        model_filename = MODEL_FILES[selected_model_name]

        model_path = MODEL_DIR / model_filename

        if not model_path.exists():

            st.error(
                f"Model file not found: {model_filename}. "
                "Please run train_models.py first."
            )

            st.stop()

        model = load_model(model_path)

        # ------------------------------------------
        # VALIDATE FEATURE COLUMNS
        # ------------------------------------------

        expected_columns = list(
            model.named_steps[
                "preprocessor"
            ].feature_names_in_
        )

        uploaded_columns = list(uploaded_df.columns)

        missing_columns = sorted(
            set(expected_columns) - set(uploaded_columns)
        )

        unexpected_columns = sorted(
            set(uploaded_columns) - set(expected_columns)
        )

        if missing_columns or unexpected_columns:

            st.error(
                "Uploaded CSV does not match "
                "the expected feature columns."
            )

            if missing_columns:
                st.write("Missing columns:", missing_columns)

            if unexpected_columns:
                st.write("Unexpected columns:", unexpected_columns)

            st.stop()

        # Maintain the same feature order used during training
        uploaded_df = uploaded_df[expected_columns]

        # ------------------------------------------
        # PREDICT BUTTON
        # ------------------------------------------

        if st.button(
            "Predict Default",
            type="primary"
        ):

            with st.spinner("Generating predictions..."):

                predictions = model.predict(uploaded_df)

                probabilities = model.predict_proba(
                    uploaded_df
                )[:, 1]

            # --------------------------------------
            # PREDICTION SUMMARY
            # --------------------------------------

            st.divider()

            st.header("Prediction Summary")

            total_customers = len(predictions)

            predicted_defaults = int(
                (predictions == 1).sum()
            )

            predicted_non_defaults = int(
                (predictions == 0).sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Customers",
                total_customers
            )

            col2.metric(
                "Predicted Defaults",
                predicted_defaults
            )

            col3.metric(
                "Predicted Non-Defaults",
                predicted_non_defaults
            )

            # --------------------------------------
            # PREDICTION DISTRIBUTION
            # --------------------------------------

            st.subheader("Prediction Distribution")

            prediction_counts = pd.Series(
                predictions
            ).value_counts().reindex(
                [1, 0],
                fill_value=0
            )

            chart_df = pd.DataFrame({
                "Prediction": ["Default", "No Default"],
                "Customers": [
                    prediction_counts.loc[1],
                    prediction_counts.loc[0]
                ]
            })

            st.bar_chart(
                chart_df.set_index("Prediction")
            )

            # --------------------------------------
            # PREDICTION RESULTS
            # --------------------------------------

            st.subheader("Prediction Results")

            results_df = uploaded_df.copy()

            results_df["Predicted Default"] = predictions

            results_df["Default Probability"] = probabilities

            results_df["Prediction Label"] = [
                "Default" if value == 1 else "No Default"
                for value in predictions
            ]

            if y_true is not None:

                results_df["Actual Default"] = y_true.values

            st.dataframe(
                results_df.head(100),
                use_container_width=True
            )

            # --------------------------------------
            # DOWNLOAD RESULTS
            # --------------------------------------

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            # --------------------------------------
            # EVALUATION METRICS
            # --------------------------------------

            st.divider()

            st.header("Model Evaluation")

            if y_true is not None:

                st.success(
                    "Actual target labels detected. "
                    "Evaluation metrics are available."
                )

                if y_true.nunique() < 2:

                    st.warning(
                        "AUC cannot be calculated because "
                        "the uploaded target contains only one class."
                    )

                    auc = None

                else:

                    auc = roc_auc_score(
                        y_true,
                        probabilities
                    )

                metrics = {
                    "Accuracy": accuracy_score(
                        y_true,
                        predictions
                    ),

                    "AUC": auc,

                    "Precision": precision_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                    "Recall": recall_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                    "F1 Score": f1_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                    "MCC": matthews_corrcoef(
                        y_true,
                        predictions
                    )
                }

                metric_columns = st.columns(3)

                for index, (metric_name, metric_value) in enumerate(
                    metrics.items()
                ):

                    display_value = (
                        "N/A"
                        if metric_value is None
                        else f"{metric_value:.4f}"
                    )

                    metric_columns[index % 3].metric(
                        metric_name,
                        display_value
                    )

                # ----------------------------------
                # CONFUSION MATRIX
                # ----------------------------------

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_true,
                    predictions,
                    labels=[0, 1]
                )

                fig, ax = plt.subplots()

                image = ax.imshow(cm)

                ax.set_title("Confusion Matrix")

                ax.set_xlabel("Predicted Label")

                ax.set_ylabel("Actual Label")

                ax.set_xticks([0, 1])

                ax.set_yticks([0, 1])

                ax.set_xticklabels(
                    ["No Default", "Default"]
                )

                ax.set_yticklabels(
                    ["No Default", "Default"]
                )

                for i in range(2):

                    for j in range(2):

                        ax.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center"
                        )

                fig.colorbar(image)

                st.pyplot(fig)

                plt.close(fig)

                # ----------------------------------
                # CLASSIFICATION REPORT
                # ----------------------------------

                st.subheader("Classification Report")

                report = classification_report(
                    y_true,
                    predictions,
                    target_names=[
                        "No Default",
                        "Default"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Evaluation metrics are not displayed "
                    "because the uploaded dataset does not "
                    "contain actual target labels."
                )

                st.info(
                    "Upload a CSV containing the "
                    "'default payment next month' column "
                    "to calculate evaluation metrics."
                )

    except Exception as error:

        st.error(
            f"An error occurred: {error}"
        )

else:

    st.info(
        "Please upload a CSV file to begin."
    )
