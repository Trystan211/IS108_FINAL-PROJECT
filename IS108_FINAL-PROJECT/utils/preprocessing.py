"""
utils/preprocessing.py
The full data cleaning and preparation pipeline sir. This runs once when the
user configures the sidebar settings, and its outputs are passed into
every subsequent phase (feature selection, training, prediction) sir.
 
Pipeline order matters sir - steps are arranged to avoid common errors:
  numeric coercion BEFORE missing-value fill (so fill uses mean, not mode,
  for columns that look categorical but are actually numeric strings).
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@st.cache_data
def full_preprocessing(df, target_col, missing_strategy,
                        apply_scaling, test_size, random_state):
    """
    Complete preprocessing pipeline sir:
      1. Convert blank strings to NaN
      2. Coerce numeric columns stored as strings
      3. Handle missing values (drop / fill / ignore)
      4. Separate features (X) and target (y)
      4.5 Drop high-cardinality ID-like columns (would explode OHE)
      5. Label-encode binary columns / OHE nominal columns
      6. Final numeric coercion pass
      7. Encode target to integers (required by ANN / MLPClassifier)
      8. Apply StandardScaler (optional but recommended for KNN & SVM)
      9. Train / test split with stratification

    Returns X_train, X_test, y_train, y_test, X_scaled, X_raw, scaler,
            target_classes, log
    """
    log = []
    df2 = df.copy()

    # Step 1: blank strings -> NaN
    # Some CSVs use whitespace " " instead of a real empty cell sir to represent
    # missing values. This converts them to proper NaN before any other logic sir.
    df2 = df2.replace(r'^\s*$', np.nan, regex=True)

    # Step 2: early numeric coercion
    # Columns like TotalCharges in the for example Telco dataset look numeric but are
    # stored as strings. pd.to_numeric converts them to float sir.
    # The 50% threshold prevents converting columns that are mostly text sir.
    for col in df2.columns:
        if col == target_col:
            continue
        converted = pd.to_numeric(df2[col], errors='coerce')
        if converted.notna().sum() > len(df2) * 0.5:
            df2[col] = converted

    # Step 3: handle missing values
    if missing_strategy == "Drop Rows":
        before = len(df2)
        df2 = df2.dropna().reset_index(drop=True)
        log.append(f"Dropped **{before - len(df2)}** rows containing missing values.")
    elif missing_strategy == "Fill with Mean (Num) / Mode (Cat)":
        for col in df2.columns:
            if df2[col].isnull().any():
                # Mean for numbers, most-frequent value for categories
                if pd.api.types.is_numeric_dtype(df2[col]):
                    df2[col] = df2[col].fillna(df2[col].mean())
                else:
                    df2[col] = df2[col].fillna(df2[col].mode()[0])
        log.append("Filled missing: mean for numeric, mode for categorical.")
    else:
        log.append("No missing value action taken.")

    # Step 4: separate features and target sir
    X = df2.drop(columns=[target_col]).copy()
    y = df2[target_col].copy()

    # Step 4.5: Drop high-cardinality identifier columns sir.
    # A categorical column where unique values > 50% of total rows is almost
    # certainly an ID field (e.g., customerID). These have zero predictive value
    # and would create thousands of dummy columns if one-hot encoded, making
    # the dataset enormous and corrupting distance/correlation calculations.
    HIGH_CARD_THRESHOLD = 0.50
    n_rows = len(X)
    dropped_id_cols = []
    cat_check = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for col in cat_check:
        if X[col].nunique() / n_rows > HIGH_CARD_THRESHOLD:
            dropped_id_cols.append(col)
    if dropped_id_cols:
        X = X.drop(columns=dropped_id_cols)
        log.append(
            f"Dropped **{len(dropped_id_cols)}** high-cardinality column(s) "
            f"(likely IDs / free text — not useful for prediction): "
            f"`{'`, `'.join(dropped_id_cols)}`"
        )

    # Step 5: Smart categorical encoding.
    # Two strategies are used depending on how many unique values a column has sir:
    #
    #   Binary (2 unique values, e.g., gender: Male/Female)
    #   → LabelEncoder: converts to 0/1. Safe because there is no false ordering sir.
    #
    #   Nominal (3+ unordered values, e.g., Contract: Month-to-month/One year/Two year)
    #   → One-Hot Encoding (get_dummies): creates one binary column per category.
    #     This prevents KNN/SVM from treating categories as if they have a numeric
    #     order (e.g., Month-to-month=0 < One year=1 would be mathematically wrong).
    le = LabelEncoder()
    label_encoded_cols = []
    ohe_cols = []

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique <= 2:
            # Binary -> safe to label-encode (0/1 carries no false ordinality) sir
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoded_cols.append(col)
        else:
            # Nominal with 3+ categories -> one-hot encode
            ohe_cols.append(col)

    if ohe_cols:
        # drop_first=False keeps all dummy columns sir (avoids ambiguity)
        # dtype=int stores them as 0/1 integers instead of True/False booleans
        X = pd.get_dummies(X, columns=ohe_cols, drop_first=False, dtype=int)
        log.append(
            f"One-Hot Encoded **{len(ohe_cols)}** nominal column(s): "
            f"`{'`, `'.join(ohe_cols)}`"
        )
    if label_encoded_cols:
        log.append(
            f"Label-encoded **{len(label_encoded_cols)}** binary column(s): "
            f"`{'`, `'.join(label_encoded_cols)}`"
        )
    if not cat_cols:
        log.append("No categorical feature columns detected.")

    # Step 6: final numeric coercion sir
    # Catches anything still non-numeric after encoding and fills leftover NaN
    # with 0 so no column can break scikit-learn's isnan check sir.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Step 7: Encode target to integers
    # MLPClassifier (ANN) strictly requires integer class labels sir.
    # KNN and SVM tolerate string labels, but encoding all three consistently
    # ensures a fair comparison and prevents hidden type-related bugs.
    le_y = None
    target_classes = None
    if not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        target_classes = list(le_y.fit(y.astype(str)).classes_)
        y = pd.Series(
            le_y.transform(y.astype(str)),
            index=y.index, name=target_col
        )
        log.append(f"Encoded target `{target_col}` to integers. Classes: {target_classes}")

    y = y.fillna(y.mode()[0])       # safety fill in case any target NaN slipped through sir

    # Save a copy of X before scaling - Phase 4 uses this for readable defaults
    X_raw = X.copy()

    # Step 8: Apply StandardScaler
    # Transforms each column to mean=0, std=1.
    # Required for KNN (Euclidean distance) and SVM (margin calculation) so
    # that large-range features don't dominate small-range ones sir.
    if apply_scaling:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )
        log.append("Applied **StandardScaler** (zero-mean, unit-variance).")
    else:
        scaler = None
        X_scaled = X.copy()
        log.append("Feature scaling not applied.")

    # Step 9: Train / test split with stratification
    # stratify=y preserves the original class ratio in both splits, which is
    # critical for imbalanced datasets so the test set reflects real distribution sir.
    # Falls back to non-stratified split if any class has too few samples.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size,
            random_state=int(random_state), stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=int(random_state)
        )
    log.append(
        f"Split → **Train:** {len(X_train)} rows | "
        f"**Test:** {len(X_test)} rows ({test_size*100:.0f}% test)."
    )

    return X_train, X_test, y_train, y_test, X_scaled, X_raw, scaler, target_classes, log