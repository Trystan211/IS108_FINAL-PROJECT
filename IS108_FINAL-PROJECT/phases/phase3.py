"""
phases/phase3.py
Phase 3 — Model Training & Evaluation.
 
This is the core engine of the application sir. It:
  1. Instantiates KNN, SVM, and ANN with the sidebar hyperparameters
  2. Optionally runs GridSearchCV to auto-tune those hyperparameters
  3. Fits all three models on the training set
  4. Evaluates them on the test set (accuracy, precision, recall, F1)
  5. Runs 5-fold cross-validation for a more robust accuracy estimate
  6. Renders the metrics table, confusion matrices, and best-model analysis
  7. Persists trained models to st.session_state so Phase 4 can use them
     without re-training on every Streamlit re-run sir
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ui.steppers import show_stepper
from utils.metrics import compute_metrics, plot_confusion_matrix


def render_phase3(
    X_train_sel, X_test_sel,
    y_train, y_test,
    X_raw, scaler, target_classes,
    selected_features,
    knn_k, svm_kernel, svm_C,
    ann_hidden_str, ann_max_iter,
    run_tuning, random_state,
):
    """
    Render Phase 3: train button -> training -> evaluation -> visualizations sir.
 
    Parameters
    ----------
    X_train_sel, X_test_sel : feature matrices filtered to selected features
    y_train, y_test         : label arrays
    X_raw                   : unscaled features (saved to session_state for Phase 4)
    scaler                  : fitted StandardScaler or None (saved for Phase 4)
    target_classes          : original class names e.g. ["No","Yes"] (for CM labels)
    selected_features       : list of column names (saved for Phase 4 form)
    knn_k                   : KNN neighbors value from sidebar
    svm_kernel, svm_C       : SVM settings from sidebar
    ann_hidden_str          : ANN hidden layer string e.g. "100" or "100,50"
    ann_max_iter            : ANN max training epochs from sidebar
    run_tuning              : bool — whether to run GridSearchCV first
    random_state            : int seed for reproducibility
    """
    # -- Stepper steps 1-3 done, steps 4 (Training) and 5 (Testing) active -------------
    show_stepper([4, 5])

    st.markdown(
        '<div class="phase-badge"> Phase 3 — Model Training &amp; Evaluation</div>',
        unsafe_allow_html=True,
    )

    train_btn = st.button("  Train All Models", type="primary", use_container_width=True)

    # Show a ready message if the button hasn't been clicked yet AND
    # no previous training result exists in session_state sir
    if not train_btn and "trained" not in st.session_state:
        st.info(" Preprocessing and feature selection ready. Click **Train All Models** to begin.")
        return

    if train_btn:
        try:
            # -- Parse ANN hidden layers ----------------------------------------
            # The user types "100" or "100,50" — we convert that string into
            # a tuple of integers that MLPClassifier expects, e.g. (100,) or (100, 50).
            try:
                hidden = tuple(int(x.strip()) for x in ann_hidden_str.split(",") if x.strip())
            except ValueError:
                hidden = (100,)
                st.warning("Invalid ANN layer format — defaulting to (100,).")

            # -- Instantiate models ---------------------------------------------
            # KNN: k=5 balances bias vs variance; minkowski at p=2 = Euclidean distance.
            # uniform weights mean all k neighbors contribute equally to the vote sir.
            knn = KNeighborsClassifier(n_neighbors=knn_k, metric="minkowski", weights="uniform")

            # SVM: class_weight="balanced" adjusts the loss for class imbalance
            # by weighting minority-class errors more heavily - important for
            # churn data where churned customers are the minority class sir.
            # probability=True enables predict_proba() for confidence scores.
            svm = SVC(kernel=svm_kernel, C=svm_C, probability=True,
                      random_state=int(random_state), class_weight="balanced")
            
            # ANN: relu avoids vanishing gradients; adam adapts learning rates
            # per-parameter; early_stopping halts training when val loss plateaus,
            # preventing overfitting without manually tuning max_iter sir.
            ann = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                max_iter=ann_max_iter,
                random_state=int(random_state),
                early_stopping=True,
                validation_fraction=0.1,        # 10% of training data used for early stopping
            )

            # -- Optional GridSearchCV auto-tuning -----------------------------
            # When run_tuning=True, each model is replaced with the best estimator
            # found by GridSearchCV before the main training loop below runs sir.
            # cv=5 uses 5-fold cross-validation; scoring="f1_macro" means the
            # search maximises F1 (not accuracy) - better for imbalanced data.
            # n_jobs=-1 uses all CPU cores to run the grid search in parallel.
            if run_tuning:
                tune_status = st.empty()
                tune_bar    = st.progress(0, text="Auto-tuning hyperparameters…")

                tune_status.write("Tuning **KNN** — searching best k…")
                knn_grid = GridSearchCV(
                    KNeighborsClassifier(metric="minkowski", weights="uniform"),
                    param_grid={"n_neighbors": [3, 5, 7, 9, 11, 15]},
                    cv=5, scoring="f1_macro", n_jobs=-1,
                )
                knn_grid.fit(X_train_sel, y_train)
                knn = knn_grid.best_estimator_          # Replace default with best found
                tune_bar.progress(0.33, text=f" KNN best k={knn_grid.best_params_['n_neighbors']}")

                tune_status.write(" Tuning **SVM** — searching best C…")
                svm_grid = GridSearchCV(
                    SVC(kernel=svm_kernel, probability=True, random_state=int(random_state)),
                    param_grid={"C": [0.1, 0.5, 1.0, 5.0, 10.0]},
                    cv=5, scoring="f1_macro", n_jobs=-1,
                )
                svm_grid.fit(X_train_sel, y_train)
                svm = svm_grid.best_estimator_
                tune_bar.progress(0.66, text=f" SVM best C={svm_grid.best_params_['C']}")

                tune_status.write(" Tuning **ANN** — searching best layer size…")
                ann_grid = GridSearchCV(
                    MLPClassifier(
                        activation="relu", solver="adam",
                        max_iter=ann_max_iter, random_state=int(random_state),
                        early_stopping=True,
                    ),
                    param_grid={"hidden_layer_sizes": [(50,), (100,), (100, 50), (150,)]},
                    cv=5, scoring="f1_macro", n_jobs=-1,
                )
                ann_grid.fit(X_train_sel, y_train)
                ann = ann_grid.best_estimator_
                tune_bar.progress(1.0, text=f" ANN best layers={ann_grid.best_params_['hidden_layer_sizes']}")
                tune_status.success(
                    f" Auto-tuning complete — "
                    f"KNN k={knn_grid.best_params_['n_neighbors']}, "
                    f"SVM C={svm_grid.best_params_['C']}, "
                    f"ANN layers={ann_grid.best_params_['hidden_layer_sizes']}"
                )
                st.divider()

            # -- Train models --------------------------------------------------
            # Fits each model on X_train_sel and immediately generates test predictions sir.
            # Results and fitted model objects are stored in dicts for use below.
            models       = {"KNN": knn, "SVM": svm, "ANN": ann}
            results      = []           # List of metric dicts - becomes the comparison table
            trained      = {}           # {name: (fitted_model, y_pred)} - used for eval + Phase 4
            progress_bar = st.progress(0, text="Starting training…")
            status       = st.empty()

            for i, (name, model) in enumerate(models.items()):
                status.write(f"Training **{name}**…")
                model.fit(X_train_sel, y_train)                 # Train on training set
                y_pred = model.predict(X_test_sel)              # Evaluate on held-out test set
                results.append(compute_metrics(name, y_test, y_pred))
                trained[name] = (model, y_pred)
                progress_bar.progress((i + 1) / 3, text=f" {name} trained.")

            status.success("All three models trained successfully!")

            # -- 5-Fold Cross-Validation ---------------------------------------
            # cross_val_score splits X_train_sel into 5 folds, trains on 4, tests
            # on 1 - repeated 5 times with different folds. The mean score is a
            # more reliable accuracy estimate than a single train/test split because
            # it averages over 5 different "test sets" from the training data sir.
            # delta shows += std, indicating how consistent performance is across folds.
            st.subheader("Cross-Validation Results (5-Fold)")
            st.caption(
                "5-fold CV splits the training data into 5 parts, trains on 4 and tests "
                "on 1 — repeated 5 times. The mean score is a more reliable performance "
                "estimate than a single split."
            )
            cv_cols = st.columns(3)
            for idx_cv, (cv_name, (cv_model, _)) in enumerate(trained.items()):
                cv_scores = cross_val_score(cv_model, X_train_sel, y_train, cv=5, scoring="accuracy")
                cv_cols[idx_cv].metric(
                    f"{cv_name} CV Accuracy",
                    f"{cv_scores.mean()*100:.2f}%",
                    delta=f"± {cv_scores.std()*100:.2f}%",      # Lower std = more consistent
                )
            st.divider()

            # -- Persist to session_state --------------------------------------
            # Streamlit re-runs the entire script on every user interaction.
            # Saving these to session_state means Phase 4 can still access them
            # after the re-run without needing to retrain from scratch sir.
            st.session_state["trained"]           = trained
            st.session_state["selected_features"] = selected_features
            st.session_state["X_raw"]             = X_raw
            st.session_state["scaler"]            = scaler
            st.session_state["target_classes"]    = target_classes

            # -- Model Persistence/ Download Trained Models (joblib) ------------
            # Saves all three trained models + scaler to disk so the session
            # can be reloaded without retraining. Users can download the file.
            import joblib, io, os
            save_payload = {
                "trained":           {n: m for n, (m, _) in trained.items()},
                "selected_features": selected_features,
                "scaler":            scaler,
                "target_classes":    target_classes,
            }
            model_buffer = io.BytesIO()
            joblib.dump(save_payload, model_buffer)
            model_buffer.seek(0)             # Reset buffer position to the start before reading
            st.download_button(
                label="Download Trained Models (.joblib)",
                data=model_buffer,
                file_name="bi_app_trained_models.joblib",
                mime="application/octet-stream",
                help="Download all three trained models + scaler. "
                     "Re-upload to skip retraining in future sessions.",
            )

            # -- Metrics Comparison Table --------------------------------------
            # highlight_max (green) and highlight_min (red) are applied per column
            # so it's immediately obvious which model wins each metric category sir.
            st.subheader("Model Performance Comparison")
            st.caption("Green = best per column  |  Red = worst  |  Measured on the test set.")
            metrics_df = pd.DataFrame(results).set_index("Model")
            st.dataframe(
                metrics_df.style
                    .highlight_max(axis=0, color="#c6efce")
                    .highlight_min(axis=0, color="#ffc7ce")
                    .format("{:.4f}"),
                use_container_width=True,
            )

            # Identify the best model by accuracy AND by F1 separately.
            # For imbalanced datasets the F1 winner is the more business-relevant choice sir.
            best    = metrics_df["Accuracy"].idxmax()
            best_f1 = metrics_df["F1-Score (Macro)"].idxmax()
            bacc    = metrics_df.loc[best, "Accuracy"]  # noqa: F841

            # -- Best Model Analysis Card --------------------------------------
            # Dynamically builds a plain-English explanation of WHY the F1 winner
            # is recommended for this business problem sir.
            # The explanation adjusts based on which model actually won.
            f1_vals = {n: metrics_df.loc[n, "F1-Score (Macro)"] for n in metrics_df.index}
            rc_vals = {n: metrics_df.loc[n, "Recall (Macro)"]   for n in metrics_df.index}

            model_descriptions = {
                "KNN": "K-Nearest Neighbors (distance-based, non-parametric)",
                "SVM": "Support Vector Machine (margin-maximising, RBF kernel)",
                "ANN": "Artificial Neural Network (multi-layer perceptron, relu/adam)",
            }
            business_best   = best_f1
            business_reason = (
                f"In classification problems with class imbalance (e.g., churn datasets where "
                f"churned customers are the minority), **Accuracy alone is misleading**. A model "
                f"that predicts 'No Churn' for every customer achieves high accuracy but zero "
                f"business value. The **F1-Score (Macro)** balances Precision and Recall equally "
                f"across all classes and is the correct metric for this problem.\n\n"
                f"**{business_best}** achieved the highest Macro F1-Score of "
                f"**{f1_vals[business_best]*100:.2f}%** and Macro Recall of "
                f"**{rc_vals[business_best]*100:.2f}%**, meaning it successfully "
                f"identified the most at-risk customers from the test set. "
                f"The {model_descriptions[business_best]} was most effective here because "
            )
            if business_best == "ANN":
                business_reason += (
                    "neural networks can learn complex, non-linear patterns in tabular customer "
                    "data through multiple layers of weighted transformations — patterns that "
                    "simpler models like KNN may miss."
                )
            elif business_best == "SVM":
                business_reason += (
                    "the RBF kernel projects the data into a higher-dimensional space where a "
                    "linear boundary separates the classes with maximum margin — highly effective "
                    "for tabular data with non-linear relationships between features."
                )
            else:
                business_reason += (
                    "the local neighborhood structure of the data is strong enough that nearby "
                    "customers in feature space tend to share the same churn outcome."
                )

            st.markdown(
                f"""
                <div class="analysis-card">
                    <h4>Model Analysis: Why {business_best} is the Best Choice for This Business Problem</h4>
                    <p>{business_reason}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider()

            # -- Confusion Matrices --------------------------------------------
            # Three subplots side by side — one per model.
            # plot_confusion_matrix() is defined in utils/metrics.py sir.
            # labels=sorted(y_test.unique()) ensures both axes use the same
            # class order across all three plots for fair visual comparison sir.
            st.subheader("Confusion Matrices")
            st.caption("Rows = True Labels  |  Columns = Predicted Labels  |  Darker diagonal = more correct predictions.")
            labels = sorted(y_test.unique())
            fig_cm, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            fig_cm.suptitle("Confusion Matrices – KNN vs SVM vs ANN", fontsize=15, fontweight="bold")
            for ax, (name, (_, y_pred)) in zip(axes, trained.items()):
                plot_confusion_matrix(ax, y_test, y_pred, name, labels)
            st.pyplot(fig_cm)
            plt.close(fig_cm)           # Free memory after rendering
            st.divider()

            # -- Per-model expanders --------------------------------------------
            # Each expander shows the model's full hyperparameter dict
            # (via model.get_params()) and its individual metric cards sir.
            # Collapsed by default to keep the page clean.
            st.subheader("Detailed Per-Model Breakdown")
            for name, (model, y_pred) in trained.items():
                with st.expander(f" {name} – Full Details"):
                    col_params, col_metrics = st.columns(2)
                    with col_params:
                        st.write("**Model Parameters:**")
                        st.json(model.get_params())         # All hyperparameters as JSON
                    with col_metrics:
                        m = compute_metrics(name, y_test, y_pred)
                        mc1, mc2 = st.columns(2)
                        mc1.metric("Accuracy",          f"{m['Accuracy']*100:.2f}%")
                        mc1.metric("Precision (Macro)", f"{m['Precision (Macro)']*100:.2f}%")
                        mc2.metric("Recall (Macro)",    f"{m['Recall (Macro)']*100:.2f}%")
                        mc2.metric("F1-Score (Macro)",  f"{m['F1-Score (Macro)']*100:.2f}%")
            st.divider()

        except Exception as e:
            # Catch-all for any unexpected crash during training.
            # Displays a friendly message with actionable fixes instead of
            # a raw Python traceback that would confuse non-technical viewers sir.
            st.error(
                f"**Model training failed.**\n\n"
                f"**Error:** `{e}`\n\n"
                f"**Common fixes:**\n"
                f"- Select the correct **Target Column** in the sidebar.\n"
                f"- Set Missing Values to **Fill with Mean/Mode** or **Drop Rows**.\n"
                f"- Make sure your dataset has more than 20 rows.\n"
                f"- Select at least one feature in Phase 2.5."
            )