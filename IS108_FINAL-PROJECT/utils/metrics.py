"""
utils/metrics.py
Evaluation helpers used by phase3.py after models are trained sir.
Kept separate so the same functions can be called for every model
without repeating code.
"""
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)


def compute_metrics(name, y_true, y_pred):
    """
    Return a dictionary of the four standard classification metrics sir.
 
    All three averaging modes (precision/recall/F1) use 'macro', which
    treats every class equally regardless of how many samples it has.
    This is the right choice for imbalanced datasets (e.g., churn) where
    the minority class is just as important as the majority class sir.
    zero_division=0 prevents a ZeroDivisionError if a class has no predictions.
    """
    return {
        "Model":             name,
        "Accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "Precision (Macro)": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall (Macro)":    round(recall_score(y_true, y_pred,    average="macro", zero_division=0), 4),
        "F1-Score (Macro)":  round(f1_score(y_true, y_pred,        average="macro", zero_division=0), 4),
    }


def plot_confusion_matrix(ax, y_true, y_pred, name, labels):
    """
    Draw a Seaborn heatmap confusion matrix on a given Matplotlib Axes object sir.
 
    Reading the matrix:
      - Rows    = True Labels  (what the answer actually was)
      - Columns = Predicted Labels (what the model said)
      - Diagonal cells (top-left to bottom-right) = correct predictions
      - Off-diagonal cells = misclassifications
 
    cmap="Blues" makes darker cells mean higher counts, so a dark diagonal
    visually signals a well-performing model at a glance.
    fmt="d" prints counts as plain integers (not scientific notation).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",      
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(f"{name} – Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)