"""
phases/phase25.py
-----------------
Phase 2.5 — Feature Selection via Pearson correlation analysis.
 
Shows two visualizations sir:
  1. Correlation heatmap  — every feature vs every other feature
  2. Bar chart            — each feature's absolute correlation with the target
 
Then provides a multiselect widget so the user can include or exclude
features before training sir. This directly addresses the "feature selection"
step in the IS 108 rubric's predictive modeling workflow checklist.
 
Returns the user's chosen features and the filtered train/test matrices.
 
PERFORMANCE NOTE:
  Computing and plotting correlations on an example dataset of X,000+ rows is slow. To prevent
  re-computing on every Streamlit re-run (e.g., user moves a slider),
  all three heavy operations - correlation matrix, heatmap figure, and
  bar chart figure - are cached in st.session_state using a lightweight
  fingerprint of the data as the cache key sir.
"""
import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# -- Cheap fingerprint (hashes shape + sample bytes, not the full array) ------
# Why not use @st.cache_data?
# @st.cache_data works well for pure functions that return simple values sir.
# Matplotlib Figure objects contain C-level state that doesn't serialise
# cleanly for Streamlit's cache, causing warnings. Storing them directly
# in st.session_state (a plain Python dict that lives in the browser session)
# is the recommended pattern for figure objects sir.
def _array_fingerprint(*arrays):
    h = hashlib.md5()
    for arr in arrays:
        h.update(str(arr.shape).encode())
        flat = arr.flat
        head = arr.flat[:min(200, arr.size)].tobytes()
        h.update(head)
    return h.hexdigest()


def _get_correlations(X_scaled, y_combined, target_col):
    """
    Return (feat_corr_matrix, target_corr) from st.session_state if the
    data hasn't changed, otherwise recompute and store. The fingerprint is
    O(1) — it only samples the first 200 elements, so cache-checking is
    effectively free even on large datasets.
    """
    # Build a unique cache key from the data fingerprint + target column name
    fp = _array_fingerprint(X_scaled.values, y_combined.values) + target_col
    key = f"_phase25_corr_{fp}"

    if key not in st.session_state:
        # Temporarily attach the target column to compute its correlations sir
        combined = X_scaled.copy()
        combined[target_col] = y_combined.values
        corr_full = combined.corr()           # Pearson correlation matrix    
        
        # Absolute correlation of each feature with the target, sorted high -> low
        target_corr = (
            corr_full[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
        )
        # Feature-vs-feature matrix (target column removed from both axes)
        feat_corr = corr_full.drop(columns=[target_col]).drop(index=[target_col])
        st.session_state[key] = (feat_corr, target_corr)

    return st.session_state[key]


def _get_heatmap(corr_matrix, heatmap_limit):
    """
    Build the correlation heatmap Matplotlib figure and cache it sir.
 
    If the dataset has more columns than heatmap_limit, only the top N most
    correlated features are shown — prevents the heatmap becoming unreadably
    small on wide datasets.
    """
    fp = _array_fingerprint(corr_matrix.values) + str(heatmap_limit)
    key = f"_phase25_heatmap_{fp}"

    if key not in st.session_state:
        capped = False
        if len(corr_matrix.columns) > heatmap_limit:
            # Keep only the top N features by maximum absolute correlation
            top_cols = (
                corr_matrix.abs().max(axis=1)
                .sort_values(ascending=False)
                .head(heatmap_limit)
                .index.tolist()
            )
            corr_matrix = corr_matrix.loc[top_cols, top_cols]
            capped = True

        # Figure size scales with the number of columns for readability sir
        fig, ax = plt.subplots(
            figsize=(
                max(8, len(corr_matrix.columns) * 0.7),
                max(6, len(corr_matrix.columns) * 0.6),
            )
        )
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, linecolor="white",
            ax=ax, annot_kws={"size": 8},
        )
        ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.session_state[key] = (fig, capped)

    return st.session_state[key]


def _get_bar_chart(target_corr):
    """
    Build the horizontal bar chart of feature-target correlations and cache it sir.
 
    The x-axis starts at 0 so bar lengths are directly comparable.
    A red dashed line at 0.10 marks the recommended minimum threshold -
    features below this line are weakly correlated and may be excluded.
    """
    fp = _array_fingerprint(target_corr.values)
    key = f"_phase25_bar_{fp}"

    if key not in st.session_state:
        # Dark blue for features above the 0.10 threshold, light blue below sir
        colors = ["#2196F3" if v >= 0.1 else "#90CAF9" for v in target_corr.values]
        fig, ax = plt.subplots(figsize=(10, max(3, len(target_corr) * 0.4)))
        ax.barh(
            target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1]
        )
        # Threshold reference line
        ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1, label="0.10 threshold")
        ax.set_xlabel("Absolute Correlation with Target")
        ax.set_title("Feature Importance by Correlation", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        st.session_state[key] = fig

    return st.session_state[key]

# ------------------------------------------------------------------------------
#  MAIN RENDER FUNCTION
# ------------------------------------------------------------------------------
 
def render_phase25(X_train, X_test, X_scaled, y_train, y_test, target_col):
    """
    Render Phase 2.5: correlation analysis and feature selector sir.
 
    Parameters
    ----------
    X_train, X_test : scaled feature matrices from Phase 2
    X_scaled        : full scaled feature matrix (train + test combined)
    y_train, y_test : label arrays from Phase 2
    target_col      : name of the target column (used as correlation axis label)
 
    Returns
    -------
    selected_features : list[str] — columns the user chose to include
    X_train_sel       : X_train filtered to selected_features only
    X_test_sel        : X_test  filtered to selected_features only
    """
    st.markdown(
        '<div class="phase-badge">Phase 2.5 — Feature Selection</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Examine feature correlations with the target and select which features "
        "to include in model training. Removing low-correlation or redundant features "
        "can improve model performance and reduce noise."
    )

    # Combine train and test labels aligned to X_scaled's index for correlation sir
    # (we need a single label array covering all rows in X_scaled)
    y_combined = pd.concat([y_train, y_test]).reindex(X_scaled.index)
    corr_matrix_df, target_corr = _get_correlations(X_scaled, y_combined, target_col)

    # -- Correlation Heatmap -------------------------------------------------------
    HEATMAP_LIMIT = 15
    with st.expander("Correlation Heatmap (all features vs target)", expanded=True):
        fig_corr, capped = _get_heatmap(corr_matrix_df, HEATMAP_LIMIT)
        if capped:
            st.caption(
                f"Dataset has many columns — heatmap shows the top {HEATMAP_LIMIT} "
                "most correlated features only. All features still appear in the bar chart below."
            )
        st.pyplot(fig_corr)

    # -- Correlation with Target Bar Chart -----------------------------------------
    st.subheader("Feature Correlation with Target")
    fig_bar = _get_bar_chart(target_corr)
    st.pyplot(fig_bar)

    st.caption(
        "🔵 Dark blue = correlation ≥ 0.10 (recommended to keep)  |  "
        "🔴 Dashed line = 0.10 threshold"
    )

    # -- Feature Selector Multiselect ---------------------------------------------
    # Default pre-selection: features with at least 0.05 absolute correlation.
    # This is a lenient threshold - features barely above zero are excluded
    # while keeping most useful columns. The user can freely override it sir.
    all_features = X_scaled.columns.tolist()
    recommended = [f for f in all_features if target_corr.get(f, 0) >= 0.05]
    if not recommended:
        recommended = all_features          # Fallback: keep everything if nothing qualifies

    selected_features = st.multiselect(
        "Select features to include in model training:",
        options=all_features,
        default=recommended,
        help=(
            "Features with higher correlation to the target are recommended. "
            "You can manually add or remove any feature before training."
        ),
    )

    # Guard: at least one feature must be selected for training to make sense sir
    if not selected_features:
        st.warning("You must select at least one feature. Defaulting to all features.")
        selected_features = all_features

    st.info(
        f"**{len(selected_features)}** of **{len(all_features)}** features selected for training."
    )

    # Filter both train and test matrices to the user's selected columns
    # These filtered versions are what Phase 3 actually trains on sir
    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]

    st.divider()

    return selected_features, X_train_sel, X_test_sel