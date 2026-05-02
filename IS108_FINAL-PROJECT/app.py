"""
================================================================================
 IS 108 – Intelligence System Final Project SY 2025-2026
 Business Intelligence Predictive Modeling Application
================================================================================
 WHAT THIS FILE DOES:
   This is the ENTRY POINT of the entire application. Think of it as the
   "main hub" - it does not contain any heavy logic itself sir. Instead, it:
     1. Configures the Streamlit page settings (tab title, layout, etc.)
     2. Builds the sidebar (file upload, model settings, preprocessing options)
     3. Calls each Phase in order, passing the results of one phase
        as inputs to the next phase.
 
 WHY THIS STRUCTURE?
   The code is split into focused modules sir. Each module handles ONE responsibility:
     - utils/     → pure Python logic (no UI): loading data, preprocessing, metrics
     - ui/        → visual helpers: CSS styling, the workflow stepper HTML
     - phases/    → each major section of the app (Phase 1, 2, 2.5, 3, 4)
 
 HOW TO RUN:
   Step 1 - Install dependencies (only needed once):
     pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
 
   Step 2 - Activate the virtual environment:
     venv\Scripts\activate (Windows)

   Step 3 - Start the app:
     streamlit run app.py
 
   Step 4 - Open your browser at:
     http://localhost:8501
 
 PROJECT FILE STRUCTURE:
   app.py                <- YOU ARE HERE SIR(entry point)
   utils/
     __init__.py         <- makes utils/ a Python package (can be empty)
     loader.py           <- load_data()          : reads CSV/Excel into DataFrame
     preprocessing.py    <- full_preprocessing() : cleans, encodes, scales, splits
     metrics.py          <- compute_metrics()    : accuracy/precision/recall/F1
                            plot_confusion_matrix(): seaborn heatmap
   ui/
     __init__.py         <- makes ui/ a Python package (can be empty)
     styles.py           <- inject_css()         : all custom CSS for the visual theme
     steppers.py         <- show_stepper()       : the 7-step workflow progress bar
   phases/
     __init__.py         <- makes phases/ a Python package (can be empty)
     phase1.py           <- render_phase1()  : Dataset Overview & Problem ID
     phase2.py           <- render_phase2()  : Data Preprocessing pipeline
     phase25.py          <- render_phase25() : Feature Selection & correlation
     phase3.py           <- render_phase3()  : Model Training & Evaluation
     phase4.py           <- render_phase4()  : Live Prediction Engine
================================================================================
"""

# -- warnings ------------------------------------------------------------------
# Python's warnings module lets us suppress non-critical messages sir.
# scikit-learn often prints ConvergenceWarnings (e.g., "ANN did not converge
# in 500 iterations") which are not errors - just informational. Suppressing
# them keeps the terminal output clean during the live demo.
import warnings

# --- streamlit ----------------------------------------------------------------
# Streamlit is the framework that turns this Python script into a web app sir.
# Every st.xxx() call (e.g., st.sidebar, st.button, st.dataframe) renders
# something directly in the browser - no HTML/JavaScript needed.
import streamlit as st

warnings.filterwarnings("ignore")

# -- Page config (must be the very first Streamlit call) -----------------------
# Important: st.set_page_config() MUST be the very first Streamlit call in
# the script - before any other st.xxx() command, and before any imports
# that themselves call Streamlit. Placing it anywhere else raises a
# StreamlitAPIException and crashes the app on startup sir.
#
# Options explained:
#   page_title            -> text shown in the browser tab
#   page_icon             -> emoji/image shown in the browser tab favicon
#   layout="wide"         -> use the full screen width (vs. narrow centered)
#   initial_sidebar_state -> whether the sidebar starts open or collapsed
st.set_page_config(
    page_title="BI Predictive Modeling App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Local imports ------------------------------------------------------------
from ui.styles import inject_css
from ui.steppers import show_stepper
from utils.loader import load_data
from phases.phase1 import render_phase1
from phases.phase2 import render_phase2
from phases.phase25 import render_phase25
from phases.phase3 import render_phase3
from phases.phase4 import render_phase4

# ------------------------------------------------------------------------------
#  MAIN FUNCTION
# ------------------------------------------------------------------------------
# Execution flow inside main() sir:
#   1. Apply CSS + render branded header
#   2. Build all sidebar widgets (upload, selectors, sliders, checkboxes)
#   3. GATE: if no file uploaded yet, show placeholder & stop with return
#   4. Call phases in order - each phase's output feeds the next
#   5. Render branded footer
def main():
    # --- Inject custom CSS theme ------------------------------------------------
    inject_css()

    # --- Branded Header ---------------------------------------------------------
    st.markdown("""
    <div class="app-header">
        <h1>Business Intelligence Predictive Modeling Application</h1>
        <p>Upload your business dataset · Configure preprocessing · Compare KNN, SVM &amp; ANN side-by-side</p>
        <span class="badge">IS 108 – Intelligence System</span>
        <span class="badge">SY 2025–2026</span>
        <span class="badge">Caraga State University</span>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    #  SIDEBAR
    # ------------------------------------------------------------------------
    # The sidebar is the collapsible left panel. We put all controls there
    # so the wide main area can stay focused on results and visualizations sir.
    # st.sidebar.xxx() is identical to st.xxx() but targets the left panel.
    st.sidebar.markdown("## ⚙️ Control Panel")
    st.sidebar.markdown("### 📁 1. Dataset Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"]
    )

    # --- No file uploaded yet ----------------------------------------------------
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            <- <strong>Get started:</strong> Upload a dataset using the sidebar to begin the predictive modeling workflow.
        </div>
        """, unsafe_allow_html=True)

        # show_stepper([1]) = highlight step 1 as "active", gray out 2–7.
        # The list parameter controls which step numbers are "active".
        show_stepper([1])
        return  # Nothing below runs until a file is uploaded sir

    # --- Load data ---------------------------------------------------------------
    # load_data() is in utils/loader.py. It is decorated with @st.cache_data,
    # so the file is only read from disk ONCE per unique file sir. On every
    # subsequent re-run (e.g., user moves a slider), the cached DataFrame
    # is returned instantly without re-parsing the file.
    # Returns an empty DataFrame on failure (error already shown to user).
    df_raw = load_data(uploaded_file)
    if df_raw.empty:
        return

    # 
    # -- Sidebar Section 2: Target Variable ----------------------------------------
    # The target column is the column the ML models will LEARN TO PREDICT.
    # Defaulting to the LAST column (index = number_of_columns - 1) follows
    # the Kaggle/UCI convention where the label is always the rightmost column sir.
    st.sidebar.markdown("### 2. Target Variable")
    target_col = st.sidebar.selectbox(
        "Select the Target Column (what to predict)",
        options=df_raw.columns.tolist(),        # Populate dropdown with all column names
        index=len(df_raw.columns) - 1,          # Pre-select the last column
    )

    # --- Preprocessing options ----------------------------------------------------
    # An st.expander wraps these widgets in a collapsible section so users
    # can hide them once configured, keeping the sidebar uncluttered sir.
    st.sidebar.markdown("### 3. Preprocessing Options")
    with st.sidebar.expander("Preprocessing Settings", expanded=True):

        # --- Missing Value Strategy ---
        # NaN (Not a Number) values break ML models if not handled sir.
        # So three strategies offered:
        #   "Drop Rows"             -> delete every row that has ANY NaN
        #   "Fill with Mean / Mode" -> replace NaN with column mean (numeric)
        #                             or most frequent value (categorical)
        #   "Do Nothing"            -> leave NaN as-is (risky - may crash models)
        missing_strategy = st.selectbox(
            "Handle Missing Values",
            ["Drop Rows", "Fill with Mean (Num) / Mode (Cat)", "Do Nothing"],
            index=0,  # Default: "Drop Rows" (safest option for most datasets)
        )

        # --- Feature Scaling Toggle ---
        # StandardScaler transforms each feature column to have mean=0, std=1.
        # WHY it matters for this project sir:
        #   KNN uses Euclidean distance — without scaling, TotalCharges (0–9000)
        #   completely overpowers Tenure (1–72) in distance calculations.
        #   SVM maximises the margin between classes — unscaled features bias
        #   the margin calculation toward high-range features.
        #   ANN converges faster with normalised inputs.
        apply_scaling = st.checkbox("Apply Feature Scaling (StandardScaler)", value=True)  # Default: ON — highly recommended for KNN and SVM
        
        # --- Train / Test Split Ratio ---
        # test_size=0.20 means 20% of rows go to the test set, 80% to training.
        # The test set is NEVER seen by the model during training - it acts as
        # a simulation of real-world unseen data to measure true performance sir.
        test_size     = st.slider("Test Set Size", 0.10, 0.50, 0.20, 0.05, format="%.2f")
        
        # --- Random State Seed ---
        # Setting a fixed seed (integer) makes all random operations in
        # train_test_split and model initialisation DETERMINISTIC - meaning
        # running the app twice with the same seed produces IDENTICAL results.
        # This is essential for reproducibility and for matching the numbers
        # in the written project documentation sir.
        random_state  = st.number_input("Random State (seed)", 0, 9999, 42)  # 42 is a common ML convention (arbitrary but memorable)

    # --- Model configuration -----------------------------------------------
    # Hyperparameters are the settings you choose BEFORE training starts.
    # Unlike model weights (which are learned from data), hyperparameters
    # control HOW the model learns. Each model has its own expander.
    st.sidebar.markdown("### 4. Model Configuration")

    with st.sidebar.expander("KNN Settings"):
        # k = the number of nearest training examples to consult when predicting.
        # At prediction time, KNN finds the k closest training points (by
        # Euclidean distance) and assigns the class that appears most often.
        # Rule of thumb: k=5 is a safe starting default.
        #   k too small (e.g., 1) -> overfits (memorises noise in training data)
        #   k too large (e.g., 20)-> underfits (too many irrelevant neighbors)
        knn_k = st.slider("Neighbors (k)", 1, 25, 5)

    with st.sidebar.expander("SVM Settings"):
        # Kernel function: transforms the feature space so SVM can find a
        # linear decision boundary in the transformed (higher-dimensional) space.
        #   "rbf"    -> Radial Basis Function — best for non-linear data (default)
        #   "linear" -> straight-line boundary — fast, works for linearly separable data
        #   "poly"   -> polynomial boundary — flexible but slower
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)

        # C = regularisation strength (the bias-variance trade-off dial).
        #   Low C  -> wider margin, tolerates more misclassifications (high bias)
        #   High C -> narrow margin, minimises misclassifications (high variance)
        #   C=1.0 is the scikit-learn default and a reasonable starting point sir.
        svm_C      = st.number_input("Regularisation (C)", 0.01, 100.0, 1.0, 0.1)

    with st.sidebar.expander("ANN Settings"):
        # hidden_layer_sizes defines the network architecture:
        #   "100"     -> one hidden layer of 100 neurons
        #   "100,50"  -> two hidden layers: 100 neurons then 50 neurons
        # More neurons = more capacity to learn complex patterns, but slower
        # training and more risk of overfitting on small datasets sir.
        ann_hidden   = st.text_input("Hidden Layer Sizes", value="100",
                                     help="e.g. '100' or '100,50' for two layers")
        
        # max_iter = maximum number of full passes over the training data.
        # Note: early_stopping (enabled inside phase3.py) will automatically
        # stop training BEFORE this limit if the validation loss plateaus,
        # preventing overfitting without needing to set a perfect max_iter sir.
        ann_max_iter = st.slider("Max Iterations", 100, 1000, 500, 50)

    with st.sidebar.expander("Auto-Tune (GridSearchCV)"):
        # GridSearchCV automates hyperparameter selection by exhaustively testing
        # every combination in a predefined parameter grid, evaluating each
        # combination via k-fold cross-validation, and selecting the best.
        # This is more principled than manual trial-and-error sir.
        # Trade-off: significantly longer runtime (~60-120 seconds extra).
        run_tuning = st.checkbox(
            "Enable Hyperparameter Tuning",
            value=False,  # Default OFF — keeps training fast for demos
            help=(
                "Runs GridSearchCV to find the best k for KNN, C for SVM, and "
                "hidden layer size for ANN. Adds ~1-2 minutes to training time."
            ),
        )
        if run_tuning:
            # Only show this caption when the checkbox is actually ticked
            st.caption(
                "Tuning tests multiple configurations via cross-validation. "
                "Allow extra time (~60–120 seconds) on larger datasets."
            )

    # ------------------------------------------------------------------------
    #  PHASES
    # ------------------------------------------------------------------------
    # Each phase function is defined in its own file inside phases/.
    # They are called sequentially because each phase's outputs feed the next sir:
    #
    #   Phase 1   -> displays dataset overview;   returns nothing
    #   Phase 2   -> preprocesses data;           returns train/test splits
    #   Phase 2.5 -> feature selection;           returns filtered feature sets
    #   Phase 3   -> trains & evaluates models;   saves to st.session_state
    #   Phase 4   -> live predictions;            reads from st.session_state
 
    # --- Phase 1: Dataset Overview & Problem Identification -------------------
    # Shows: first-5-rows preview, row/column/null count metrics,
    # class distribution bar chart, column dtype table, and the
    # "Business Problem Identification" expander with fillable fields.
    # Takes df_raw, target_col, and the uploaded_file object (for the filename)sir.
    # Returns nothing - purely displays UI. 
    render_phase1(df_raw, target_col, uploaded_file)

    # --- Phase 2 — Preprocessing ------------------------------------------------
    # Runs the full data cleaning and preparation pipeline using the
    # sidebar settings collected above.
    #
    # Returns a tuple unpacked into these variables sir:
    #   pipeline_ok    -> True if preprocessing succeeded without errors
    #   X_train        -> training feature matrix (scaled if apply_scaling=True)
    #   X_test         -> test feature matrix
    #   y_train        -> training labels (integer-encoded)
    #   y_test         -> test labels (integer-encoded)
    #   X_scaled       -> full feature matrix (train+test combined, scaled)
    #                     used by Phase 2.5 to compute correlations
    #   X_raw          -> full feature matrix BEFORE scaling
    #                     used by Phase 4 to show human-readable defaults
    #   scaler         -> the fitted StandardScaler (or None if scaling is off)
    #                     used by Phase 4 to scale new prediction inputs
    #   target_classes -> original class names e.g. ["No","Yes"]
    #                     used by Phase 3 confusion matrix axis labels
    (pipeline_ok, X_train, X_test,
     y_train, y_test,
     X_scaled, X_raw,
     scaler, target_classes) = render_phase2(
        df_raw, target_col, missing_strategy,
        apply_scaling, test_size, int(random_state)
    )

    # --- Preprocessing Gate ----------------------------------------------------
    # If preprocessing failed (e.g., all rows dropped, incompatible data types),
    # render_phase2() already showed an st.error() with the reason.
    # We simply stop here - no point running feature selection or training
    # on data that wasn't successfully processed sir.
    if not pipeline_ok:
        return

    # --- Phase 2.5 — Feature Selection -----------------------------------------
    # Generates two visualizations of feature-target relationships:
    #   1. Pearson correlation heatmap (all features vs all features)
    #   2. Horizontal bar chart of absolute correlation with target
    # Then shows a multiselect widget so the user can include/exclude features.
    #
    # Returns:
    #   selected_features -> list of column names the user chose to keep
    #   X_train_sel       -> X_train filtered to selected columns
    #   X_test_sel        -> X_test  filtered to selected columns
    #
    # Filtering here means Phase 3 trains ONLY on the chosen features sir.
    # Removing low-correlation or noisy features often improves model accuracy.
    selected_features, X_train_sel, X_test_sel = render_phase25(
        X_train, X_test, X_scaled, y_train, y_test, target_col
    )

    # --- Phase 3 — Training & Evaluation ---------------------------------------
    # The core machine learning engine. When "Train All Models" is clicked:
    #
    #   OPTIONAL SIR: GridSearchCV finds best hyperparameters (if run_tuning=True)
    #
    #   Training:
    #     KNN  -> KNeighborsClassifier(n_neighbors=knn_k, metric="minkowski")
    #     SVM  -> SVC(kernel=svm_kernel, C=svm_C, probability=True)
    #     ANN  -> MLPClassifier(hidden_layer_sizes, relu, adam, early_stopping)
    #
    #   Evaluation (all computed on the held-out test set):
    #     - Accuracy, Precision (macro), Recall (macro), F1-Score (macro)
    #     - 5-fold cross-validation accuracy on the training set
    #     - Side-by-side Seaborn confusion matrix heatmaps
    #     - Per-model detail expanders (hyperparams + individual metrics)
    #     - "Best Model" analysis card explaining WHY one model is recommended
    #
    #   After training: saves models + metadata to st.session_state so
    #   Phase 4 can make predictions without re-training on every re-run sir.
    render_phase3(
        X_train_sel, X_test_sel,                 # Feature matrices (post feature selection)
        y_train, y_test,                         # Label arrays
        X_raw, scaler, target_classes,           # Needed for Phase 4 prediction form
        selected_features,                       # Column list saved to session_state
        knn_k, svm_kernel, svm_C,                # KNN and SVM hyperparameters from sidebar
        ann_hidden, ann_max_iter,                # ANN hyperparameters from sidebar
        run_tuning, random_state,                # Auto-tune flag + reproducibility seed
    )

    # --- Phase 4 — Prediction Output ------------------------------------------
    # Only renders if Phase 3 has completed at least once (checked internally
    # via `"trained" in st.session_state`).
    #
    # Reads from st.session_state:
    #   "trained"           -> dict of {model_name: fitted model object}
    #   "selected_features" -> list of feature column names for the form
    #   "X_raw"             -> unscaled feature matrix (for default input values)
    #   "scaler"            -> fitted scaler (to scale new inputs before predicting)
    #   "target_classes"    -> class label names (to decode integer predictions)
    #
    # The prediction form renders one number_input per selected feature.
    # On submit: scales the input, queries all 3 models, displays individual
    # predictions and a majority-vote final verdict sir.
    render_phase4()

    # --- Footer ------------------------------------------------------------------
    st.divider()
    st.markdown("""
    <div style="text-align:center; padding: 12px; background:#F0F7FF;
                border-radius:8px; border:1px solid #BBDEFB;
                font-size:0.82rem; color:#1565C0;">
        <strong>IS 108 – Intelligence System Final Project SY 2025–2026</strong><br>
        Caraga State University · College of Computing and Information Sciences<br>
        Tristan Rhyl C. Penaso · Airah Nichole Montillano · Kylle Mae Mercado
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()