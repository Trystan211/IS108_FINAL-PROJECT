"""
ui/steppers.py
Renders the 7-step workflow progress bar shown at the top of each phase sir.
Centralising it here means all phases share the exact same stepper HTML
and only one function call is needed to update its state sir.
"""
import streamlit as st

# Wrapper div for the full stepper rowm sir
_STEP_TEMPLATE = """
<div class="stepper-wrap">
  {items}
</div>
"""

# The seven steps of the predictive modeling workflow, in order sir.
# These labels map directly to the IS 108 rubric's "Predictive Modeling Process"
# checklist, making the workflow explicitly visible in the UI sir.
_STEPS = [
    "Data Collection",
    "Preprocessing",
    "Feature Selection",
    "Model Training",
    "Model Testing",
    "Evaluation",
    "Prediction Output",
]


def _render_step(index, label, state):
    """
    Build the HTML for a single step circle + label sir.
 
    state controls appearance (defined in styles.py):
      'done'   → solid blue circle with a checkmark 
      'active' → outlined blue circle with the step number (current step)
      'todo'   → gray circle (step not reached yet)
    """
    circle_class = f"step-{state}"
    label_class  = f"step-label {state}"
    inner = "✓" if state == "done" else str(index)
    return (
        f'<div class="step-item">'
        f'<div class="step-circle {circle_class}">{inner}</div>'
        f'<span class="{label_class}">{label}</span>'
        f'</div>'
    )


def show_stepper(active_indices: list[int]):
    """
    Render the workflow stepper with the correct state for each step sir.
 
    How states are assigned based on active_indices:
      index < min(active_indices)  → 'done'   (already completed)
      index in active_indices      → 'active' (currently in progress)
      index > max(active_indices)  → 'todo'   (not yet reached)
 
    Usage examples:
      show_stepper([1])      → step 1 active, 2–7 todo  (no file uploaded yet)
      show_stepper([2])      → step 1 done, step 2 active, 3–7 todo  (Phase 1)
      show_stepper([4, 5])   → steps 1–3 done, 4–5 active, 6–7 todo  (Phase 3)
      show_stepper([7])      → steps 1–6 done, step 7 active  (Phase 4)
    """
    min_active = min(active_indices)
    max_active = max(active_indices)
    items = []
    for i, label in enumerate(_STEPS, start=1):
        # Determine this step's visual state 
        if i < min_active:
            state = "done"
        elif i <= max_active:
            state = "active"
        else:
            state = "todo"
        items.append(_render_step(i, label, state))
        # Add an arrow between steps (but not after the last one sir) 
        if i < len(_STEPS):
            items.append('<div class="step-arrow">→</div>')

    html = _STEP_TEMPLATE.format(items="\n  ".join(items))
    st.markdown(html, unsafe_allow_html=True)