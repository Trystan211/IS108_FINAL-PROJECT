"""
phases/phase2.py
----------------
Phase 2 — Data Preprocessing.
 
Calls full_preprocessing() from utils/preprocessing.py using the sidebar
settings, then displays an expandable report sir showing:
  - A log of every step that was applied (drop rows, encode, scale, etc.)
  - Side-by-side raw vs processed data preview
 
Returns all the outputs from the pipeline so the next phases can use them.
If preprocessing fails, returns pipeline_ok=False and all other values as None sir.
"""
import streamlit as st

from utils.preprocessing import full_preprocessing


def render_phase2(df_raw, target_col, missing_strategy,
                  apply_scaling, test_size, random_state):
    """
    Run the preprocessing pipeline and display its report sir.
 
    Parameters
    ----------
    df_raw           : raw DataFrame from Phase 1
    target_col       : column name being predicted
    missing_strategy : user's choice for handling NaN values
    apply_scaling    : bool — whether to apply StandardScaler
    test_size        : float — fraction of data for the test set
    random_state     : int — seed for reproducibility
 
    Returns
    -------
    pipeline_ok    : bool — False if preprocessing threw an exception
    X_train        : training feature matrix
    X_test         : test feature matrix
    y_train        : training labels
    y_test         : test labels
    X_scaled       : full scaled feature matrix (used by Phase 2.5 for correlations)
    X_raw          : full unscaled feature matrix (used by Phase 4 for input defaults)
    scaler         : fitted StandardScaler or None
    target_classes : original class name list e.g. ["No","Yes"], or None
    """
    st.markdown(
        '<div class="phase-badge">Phase 2 — Data Preprocessing</div>',
        unsafe_allow_html=True,
    )

    # Run the pipeline - all the cleaning, encoding, scaling, and splitting
    # logic lives in utils/preprocessing.py, not here. This phase only
    # handles calling it and displaying the results sir.
    try:
        (X_train, X_test, y_train, y_test,
         X_scaled, X_raw, scaler,
         target_classes, prep_log) = full_preprocessing(
            df_raw, target_col, missing_strategy,
            apply_scaling, test_size, int(random_state)
        )
        pipeline_ok = True
    except Exception as e:
        # Show a friendly error and return a failure tuple.
        # All None values signal to app.py that it should stop (pipeline_ok=False).
        st.error(f"Preprocessing failed: {e}")
        return False, None, None, None, None, None, None, None, None

    # -- Preprocessing Report ------------------------------------------------
    # Collapsed by default (expanded=False) so it doesn't dominate the screen,
    # but can be opened to inspect exactly what happened to the data sir.
    with st.expander("View Preprocessing Steps & Processed Data", expanded=False):

        # prep_log is a list of plain-English strings built inside preprocessing.py
        # describing every action taken (e.g., "Dropped 11 rows", "Scaled features")
        st.write("**Steps Applied:**")
        for step in prep_log:
            st.markdown(f"- {step}")

        # Side-by-side comparison lets the user see exactly what changed sir
        st.write("**Raw vs Processed Data (first 5 rows):**")
        raw_col, proc_col = st.columns(2)
        with raw_col:
            st.caption("Raw (original)")
            st.dataframe(df_raw.head(), use_container_width=True)
        with proc_col:
            st.caption("Processed (encoded + scaled)")
            st.dataframe(X_scaled.head(), use_container_width=True)

    st.divider()

    return (pipeline_ok, X_train, X_test, y_train, y_test,
            X_scaled, X_raw, scaler, target_classes)