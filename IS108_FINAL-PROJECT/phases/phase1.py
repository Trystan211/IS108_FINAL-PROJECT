"""
phases/phase1.py
----------------
Phase 1 — Dataset Overview & Problem Identification.
 
Displays the first look at the uploaded dataset:
  - Business problem identification form (fillable fields for the demo)
  - First 5 rows preview
  - Row / column / missing-cell counts
  - Target class distribution bar chart
  - Column dtype summary table
 
Returns nothing - this phase is purely informational display sir.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from ui.steppers import show_stepper


def render_phase1(df_raw, target_col, uploaded_file):
    """
    Render Phase 1: dataset preview, class distribution, and problem form.
 
    Parameters
    ----------
    df_raw        : raw DataFrame straight from the file uploader
    target_col    : the column the user selected as the prediction target
    uploaded_file : the Streamlit UploadedFile object (used for its .name)
    """
    # Stepper: step 1 (Data Collection) is now done -> step 2 (Preprocessing) active
    show_stepper([2])

    st.markdown(
        '<div class="phase-badge">Phase 1 — Dataset Overview &amp; Problem Identification</div>',
        unsafe_allow_html=True,
    )

    # --- Business Problem Identification ---------------------------------------
    # This expander satisfies the rubric's "Problem Identification" requirement
    # The text inputs are purely informational - they do not affect any downstream ML processing sir.
    with st.expander("Business Problem Identification", expanded=True):
        st.markdown(
            f"""
            **Problem Type:** Classification - predicting the value of `{target_col}`
 
            **Business Context:**
            This application addresses a supervised machine learning problem where the goal
            is to predict **`{target_col}`** based on the other features in the dataset.
            Accurate predictions enable businesses to take proactive decisions — for example,
            identifying customers likely to churn, employees at risk of attrition, or
            transactions likely to be fraudulent — before the outcome occurs.
 
            **Predictive Modeling Workflow Applied:**
            1. Data Collection & Loading
            2. Exploratory Data Analysis
            3. Data Preprocessing (missing values, encoding, scaling)
            4. Feature Selection (correlation analysis)
            5. Model Training — KNN, SVM, ANN
            6. Model Evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix
            7. Prediction on New Input
            """
        )
        st.divider()
        # Fillable dataset info fields sir
        st.markdown("**Dataset Information**")
        di_col1, di_col2 = st.columns(2)
        with di_col1:
            st.text_input(
                "Dataset Name / Source",
                value=uploaded_file.name,        # Pre-filled with the uploaded filename sir
                help="Name or origin of the dataset (e.g. Telco Customer Churn – Kaggle)",
                key="ds_source",
            )
        with di_col2:
            st.text_input(
                "Business Problem Being Solved",
                placeholder="e.g. Predict which customers will churn next month",
                help="Describe in one sentence what this model is solving",
                key="ds_problem",
            )
        st.text_area(
            "Dataset Description (optional)",
            placeholder=(
                "Describe the dataset: where it came from, how many records, "
                "what each row represents, time period covered, etc."
            ),
            height=80,
            key="ds_description",
        )

    # --- Data Preview ---------------------------------------------------------
    # Shows only the first 5 rows lang ni sir - enough for a quick sanity check without
    # overwhelming the screen with thousands of rows sir.
    st.subheader("Data Preview (First 5 Rows)")
    st.dataframe(df_raw.head(), use_container_width=True)

    # Three summary metrics displayed side by side
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",    f"{df_raw.shape[0]:,}")
    c2.metric("Total Columns", f"{df_raw.shape[1]:,}")
    c3.metric("Missing Cells", f"{df_raw.isnull().sum().sum():,}")   # total NaN count ni sir

    # --- Class Distribution -----------------------------------------------------
    # A bar chart of how many rows belong to each class in the target column sir.
    # This immediately reveals class imbalance (e.g., far more "No Churn" than
    # "Churn"), which is important context for explaining why we use macro F1
    # instead of plain accuracy sir.
    st.subheader(f"Class Distribution — `{target_col}`")
    col_chart, col_dtypes = st.columns(2)

    with col_chart:
        vc = df_raw[target_col].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=vc.index.astype(str), y=vc.values, palette="Blues_d", ax=ax)
        ax.set_title("Target Class Counts", fontsize=11)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_dtypes:
        st.write("**Column Data Types**")
        import pandas as pd
        dtype_df = pd.DataFrame({
            "Column":   df_raw.dtypes.index,
            "Dtype":    df_raw.dtypes.values.astype(str),
            "Non-Null": df_raw.notna().sum().values,
            "Null":     df_raw.isna().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.divider()