import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# --------------------------------------------------
# Paths & data loading
# --------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@st.cache_data
def load_csv_safe(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path.name}: {e}")
        return None


d1_path = PROCESSED_DIR / "processed_D1_defects.csv"
d4_path = PROCESSED_DIR / "processed_D4_cicd.csv"
d5_path = PROCESSED_DIR / "processed_D5_sprints.csv"

df_d1 = load_csv_safe(d1_path)
df_d4 = load_csv_safe(d4_path)
df_d5 = load_csv_safe(d5_path)


# --------------------------------------------------
# Generic ML helper
# --------------------------------------------------

def train_basic_classifier(df, label_col, drop_cols=None, test_size=0.2):
    if df is None or label_col not in df.columns:
        return None, None

    df = df.dropna(subset=[label_col])
    if df.empty:
        return None, None

    if drop_cols is None:
        drop_cols = []

    drop_cols = set(drop_cols + [label_col, "source_dataset"])
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        return None, None

    X = df[feature_cols]
    y = df[label_col]

    # If there is only one class (e.g. all 0), skip training
    if y.nunique() < 2:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
        "features": feature_cols,
        "importances": getattr(model, "feature_importances_", None),
    }

    if hasattr(model, "predict_proba") and y.nunique() == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            metrics["auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["auc"] = None
        metrics["y_prob"] = y_prob
    else:
        metrics["auc"] = None

    return model, metrics

# --------------------------------------------------
# High-level KPIs
# --------------------------------------------------

def compute_kpis():
    kpis = {}

    # Reliability: D1 defect rate
    if df_d1 is not None and "reliability_target_defect" in df_d1.columns:
        d1_non_null = df_d1.dropna(subset=["reliability_target_defect"])
        if not d1_non_null.empty:
            defect_rate = d1_non_null["reliability_target_defect"].mean()
            kpis["defect_rate"] = defect_rate
            kpis["defect_samples"] = len(d1_non_null)

    # Risk: D4 failure rate
    if df_d4 is not None and "is_failed_pipeline" in df_d4.columns:
        d4_non_null = df_d4.dropna(subset=["is_failed_pipeline"])
        if not d4_non_null.empty:
            failure_rate = d4_non_null["is_failed_pipeline"].mean()
            kpis["failure_rate"] = failure_rate
            kpis["failure_samples"] = len(d4_non_null)

    # Speed: D5 underperforming sprint rate
    if df_d5 is not None and "underperforming_sprint" in df_d5.columns:
        d5_non_null = df_d5.dropna(subset=["underperforming_sprint"])
        if not d5_non_null.empty:
            under_rate = d5_non_null["underperforming_sprint"].mean()
            kpis["underperforming_rate"] = under_rate
            kpis["under_samples"] = len(d5_non_null)

    return kpis


# --------------------------------------------------
# Streamlit App Layout
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Enhanced FinTech Project Dashboard",
    layout="wide",
)

st.title("AI-Enhanced Software Project Management Dashboard (FinTech)")
st.caption(
    "Visualizing Speed, Reliability, and Risk Indicators using Machine Learning on Secondary Datasets."
)

kpis = compute_kpis()

# Top KPI row
st.subheader("Overview: Project Health Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    if "defect_rate" in kpis:
        st.metric(
            "Defect Rate (Reliability)",
            f"{kpis['defect_rate']*100:.1f} %",
            help=f"Based on {kpis['defect_samples']} modules from D1.",
        )
    else:
        st.metric("Defect Rate (Reliability)", "N/A")

with col2:
    if "failure_rate" in kpis:
        st.metric(
            "CI/CD Failure Rate (Risk)",
            f"{kpis['failure_rate']*100:.1f} %",
            help=f"Based on {kpis['failure_samples']} pipelines from D4.",
        )
    else:
        st.metric("CI/CD Failure Rate (Risk)", "N/A")

with col3:
    if "underperforming_rate" in kpis:
        st.metric(
            "Underperforming Sprints (Speed)",
            f"{kpis['underperforming_rate']*100:.1f} %",
            help=f"Based on {kpis['under_samples']} sprints from D5.",
        )
    else:
        st.metric("Underperforming Sprints (Speed)", "N/A")


tab_reliability, tab_risk, tab_speed = st.tabs(
    ["🔵 Reliability (Defects - D1)", "🟠 Risk (CI/CD - D4)", "🟢 Speed (Sprints - D5)"]
)

# --------------------------------------------------
# Reliability Tab (D1)
# --------------------------------------------------

with tab_reliability:
    st.header("Reliability Modelling – Defect Prediction (D1)")

    if df_d1 is None or "reliability_target_defect" not in df_d1.columns:
        st.warning("D1 dataset not available or target column missing.")
    else:
        df = df_d1.dropna(subset=["reliability_target_defect"])
        st.write("Sample of reliability dataset (D1):")
        st.dataframe(df.head())

        # Class distribution
        st.subheader("Class Distribution")
        class_counts = df["reliability_target_defect"].value_counts().rename({0: "Non-defective", 1: "Defective"})
        fig = px.bar(
            class_counts,
            x=class_counts.index,
            y=class_counts.values,
            labels={"x": "Class", "y": "Count"},
            title="Defective vs Non-Defective Modules",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Train model
        st.subheader("Random Forest Reliability Model")
        model, metrics = train_basic_classifier(df, "reliability_target_defect")

        if model is None:
            st.warning("Could not train reliability model (possibly only one class present).")
        else:
            acc = metrics["accuracy"]
            auc = metrics.get("auc", None)
            st.write(f"**Accuracy:** {acc:.3f}")
            if auc is not None:
                st.write(f"**AUC-ROC:** {auc:.3f}")

            # Confusion matrix heatmap
            cm = metrics["confusion_matrix"]
            labels = sorted(df["reliability_target_defect"].unique())
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                title="Confusion Matrix – Reliability Model",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance
            if metrics["importances"] is not None:
                st.subheader("Feature Importance")
                feat_df = pd.DataFrame(
                    {
                        "feature": metrics["features"],
                        "importance": metrics["importances"],
                    }
                ).sort_values("importance", ascending=False).head(15)
                fig_imp = px.bar(
                    feat_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Predictors of Defective Modules",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importances are not available for this model.")


# --------------------------------------------------
# Risk Tab (D4 – CI/CD)
# --------------------------------------------------

with tab_risk:
    st.header("Risk Modelling – CI/CD Pipeline Failures (D4)")

    if df_d4 is None or "is_failed_pipeline" not in df_d4.columns:
        st.warning("D4 dataset not available or target column missing.")
    else:
        df = df_d4.dropna(subset=["is_failed_pipeline"])
        st.write("Sample of CI/CD dataset (D4):")
        st.dataframe(df.head())

        st.subheader("Failure Distribution")
        fail_counts = df["is_failed_pipeline"].value_counts().rename({0: "Success", 1: "Failure"})
        fig = px.bar(
            fail_counts,
            x=fail_counts.index,
            y=fail_counts.values,
            labels={"x": "Pipeline Status", "y": "Count"},
            title="CI/CD Success vs Failure",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Duration vs failure scatter
        if "pipeline_duration_minutes" in df.columns:
            st.subheader("Pipeline Duration vs Failure")
            fig_sc = px.scatter(
                df,
                x="pipeline_duration_minutes",
                y="is_failed_pipeline",
                labels={"pipeline_duration_minutes": "Duration (min)", "is_failed_pipeline": "Failed"},
                title="CI/CD Duration vs Failure",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Random Forest Risk Model")
        model, metrics = train_basic_classifier(df, "is_failed_pipeline")

        if model is None:
            st.warning("Could not train risk model (possibly only one class present).")
        else:
            acc = metrics["accuracy"]
            auc = metrics.get("auc", None)
            st.write(f"**Accuracy:** {acc:.3f}")
            if auc is not None:
                st.write(f"**AUC-ROC:** {auc:.3f}")

            cm = metrics["confusion_matrix"]
            labels = sorted(df["is_failed_pipeline"].unique())
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                title="Confusion Matrix – CI/CD Risk Model",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            if metrics["importances"] is not None:
                st.subheader("Feature Importance")
                feat_df = pd.DataFrame(
                    {
                        "feature": metrics["features"],
                        "importance": metrics["importances"],
                    }
                ).sort_values("importance", ascending=False).head(15)
                fig_imp = px.bar(
                    feat_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Predictors of CI/CD Failure",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importances are not available for this model.")


# --------------------------------------------------
# Speed Tab (D5 – Sprints)
# --------------------------------------------------

with tab_speed:
    st.header("Speed Modelling – Sprint Performance (D5)")

    if df_d5 is None or "underperforming_sprint" not in df_d5.columns:
        st.warning("D5 dataset not available or target column missing.")
    else:
        df = df_d5.dropna(subset=["underperforming_sprint"])
        st.write("Sample of sprint dataset (D5):")
        st.dataframe(df.head())

        st.subheader("Underperforming vs Normal Sprints")
        under_counts = df["underperforming_sprint"].value_counts().rename({0: "Normal", 1: "Underperforming"})
        fig = px.bar(
            under_counts,
            x=under_counts.index,
            y=under_counts.values,
            labels={"x": "Sprint Type", "y": "Count"},
            title="Sprint Performance Classes",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "sprint_velocity_ratio" in df.columns:
            st.subheader("Sprint Velocity Distribution")
            fig_hist = px.histogram(
                df,
                x="sprint_velocity_ratio",
                nbins=20,
                title="Sprint Velocity Ratio Distribution",
                labels={"sprint_velocity_ratio": "Completed / Planned Story Points"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Random Forest Speed Model")
        model, metrics = train_basic_classifier(df, "underperforming_sprint")

        if model is None:
            st.warning("Could not train speed model (possibly only one class present).")
        else:
            acc = metrics["accuracy"]
            auc = metrics.get("auc", None)
            st.write(f"**Accuracy:** {acc:.3f}")
            if auc is not None:
                st.write(f"**AUC-ROC:** {auc:.3f}")

            cm = metrics["confusion_matrix"]
            labels = sorted(df["underperforming_sprint"].unique())
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                title="Confusion Matrix – Sprint Performance Model",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            if metrics["importances"] is not None:
                st.subheader("Feature Importance")
                feat_df = pd.DataFrame(
                    {
                        "feature": metrics["features"],
                        "importance": metrics["importances"],
                    }
                ).sort_values("importance", ascending=False).head(15)
                fig_imp = px.bar(
                    feat_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Predictors of Underperforming Sprints",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importances are not available for this model.")
