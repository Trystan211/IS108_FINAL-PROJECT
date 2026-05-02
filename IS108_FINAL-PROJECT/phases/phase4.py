"""
phases/phase4.py
Phase 4 — Live Prediction Engine.
 
Reads the models trained in Phase 3 from st.session_state and lets the
user enter new feature values to get a real-time prediction from all three
models simultaneously, followed by a majority-vote verdict sir.
 
This phase only renders if Phase 3 has been completed at least once in
the current browser session (checked via "trained" in st.session_state) sir.
"""
from collections import Counter

import pandas as pd
import streamlit as st

from ui.steppers import show_stepper


def render_phase4():
    """
    Render Phase 4: feature input form -> predictions -> majority vote.
 
    Reads from st.session_state sir (populated by Phase 3):
      "trained"           → {model_name: (fitted_model, y_pred_from_training)}
      "selected_features" → list of feature column names for the input form
      "X_raw"             → unscaled feature matrix (for human-readable defaults)
      "scaler"            → fitted StandardScaler or None
      "target_classes"    → original class labels e.g. ["No","Yes"] or None sir
    """
    # Guard: silently do nothing if Phase 3 hasn't run yet
    # This prevents the Phase 4 form from appearing before any models exist sir.
    if "trained" not in st.session_state:
        return

    # Stepper: all steps 1-6 done, step 7 (Prediction Output) active sir
    show_stepper([7])

    st.markdown(
        '<div class="phase-badge"> Phase 4 — Prediction Output</div>',
        unsafe_allow_html=True,
    )

    # Unpack everything Phase 3 saved to session_state
    _trained           = st.session_state["trained"]
    _selected_features = st.session_state["selected_features"]
    _X_raw             = st.session_state["X_raw"]
    _scaler            = st.session_state["scaler"]
    _target_classes    = st.session_state["target_classes"]

    st.subheader(" Live Prediction Engine")
    st.caption(
        "Enter real feature values below and click **Predict** to get a live "
        "prediction from all three trained models."
    )

    # -- Input Form -----------------------------------------------------------
    # st.form() groups all inputs together so the app only re-runs when the
    # user clicks the submit button, not on every individual keypress sir.
    # Up to 4 inputs per row (chunk=4) keeps the form readable.
    with st.form("pred_form"):
        input_vals = {}
        chunk = 4
        rows = [_selected_features[i:i+chunk] for i in range(0, len(_selected_features), chunk)]
        for row in rows:
            pcols = st.columns(len(row))
            for pc, col in zip(pcols, row):
                # Default value = unscaled column mean from training data
                # Using X_raw (unscaled) keeps the defaults human-readable sir
                # (e.g., tenure = 32.4 months, not the standardised -0.12).
                raw_mean = float(_X_raw[col].mean()) if col in _X_raw.columns else 0.0
                input_vals[col] = pc.number_input(
                    col,
                    value=round(raw_mean, 4),
                    format="%.4f",
                )
        predict_btn = st.form_submit_button("Predict", use_container_width=True)

    # Nothing to do until the user clicks Predict sir
    if not predict_btn:
        return

    try:
        # -- Input Validation --------------------------------------------------
        # Check if any entered value is more than 5 standard deviations from
        # the training mean sir. Such extreme outliers may produce unreliable
        # predictions and are worth flagging before showing results.
        validation_warnings = []
        for col in _selected_features:
            if col in _X_raw.columns:
                col_mean = _X_raw[col].mean()
                col_std  = _X_raw[col].std()
                val      = input_vals[col]
                if col_std > 0 and abs(val - col_mean) > 5 * col_std:
                    validation_warnings.append(
                        f"**{col}** = `{val:.4f}` is far outside the normal range "
                        f"(mean: {col_mean:.2f}, std: {col_std:.2f}). "
                        f"This may produce an unreliable prediction."
                    )
        if validation_warnings:
            st.warning(
                "**Unusual input values detected:**\n\n"
                + "\n\n".join(f"- {w}" for w in validation_warnings)
            )

        # Build a single-row DataFrame from the user's input values sir,
        input_df = pd.DataFrame([input_vals])[_selected_features]

        # -- Scale input using the same scaler fitted during training ----------
        # The models were trained on scaled data, so new inputs MUST be scaled
        # with the same parameters sir (mean and std from training).
        # Fitting a new scaler on the single input row would give wrong results sir.
        # So we reconstruct the full column list the scaler knows about, fill any
        # missing columns with 0, scale, then slice back to selected_features.
        if _scaler is not None:
            all_cols = (
                _scaler.feature_names_in_
                if hasattr(_scaler, "feature_names_in_")
                else _selected_features
            )
            full_input  = pd.DataFrame([{c: input_vals.get(c, 0.0) for c in all_cols}])
            full_scaled = pd.DataFrame(_scaler.transform(full_input), columns=all_cols)
            input_scaled = full_scaled[_selected_features]       # Keep only selected features
        else:
            input_scaled = input_df                              # No scaler - use raw values directly

        # -- Individual Model Predictions --------------------------------------
        # Each model returns an integer sir (0 or 1 for binary classification).
        # target_classes maps that back to the original label e.g. 0 → "No", 1 → "Yes".
        st.write("---")
        st.subheader("Prediction Results")

        predictions = {}
        pcols = st.columns(3)
        for idx, (mname, (mobj, _)) in enumerate(_trained.items()):
            raw_pred = mobj.predict(input_scaled)[0]
            if _target_classes is not None:
                try:
                    label = _target_classes[int(raw_pred)]       # Decode integer back to class name
                except (IndexError, ValueError):
                    label = str(raw_pred)
            else:
                label = str(raw_pred)
            predictions[mname] = label
            pcols[idx].metric(f"{mname} Prediction", label)

        # -- Majority Vote -----------------------------------------------------
        # Counter tallies how many models predicted each class.
        # most_common(1)[0][0] returns the class with the highest vote count sir.
        # Three tiers of confidence:
        #   3/3 agree → High confidence  (green success box)
        #   2/3 agree → Moderate         (amber warning box)
        #   0/3 agree → No majority      (red error box — rare with 3 models)
        st.write("")
        vote_counts = Counter(predictions.values())
        majority    = vote_counts.most_common(1)[0][0]
        total       = len(predictions)
        agree_count = vote_counts[majority]

        if agree_count == total:
            st.success(f"**All 3 models agree: `{majority}`** — High confidence prediction.")
        elif agree_count == 2:
            minority = [k for k, v in predictions.items() if v != majority]
            st.warning(
                f"**Majority vote: `{majority}`** (2 out of 3 models). "
                f"{minority[0]} disagrees — moderate confidence."
            )
        else:
            st.error("All models disagree — no clear majority. Consider retraining or adjusting features.")

    except Exception as pred_err:
        st.error(f"Prediction failed: {pred_err}")