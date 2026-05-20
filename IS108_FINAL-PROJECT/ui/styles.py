"""
ui/styles.py
All custom CSS for the BI Predictive Modeling App sir.
Keeping every style rule in one file means design changes only need
to be made here — no hunting through multiple phase files sir.
 
Call inject_css() once at the very start of main() in app.py sir.
The CSS is injected via st.markdown() with unsafe_allow_html=True,
which pushes a raw <style> block into the page's HTML head.
"""
import streamlit as st

_CSS = """
<style>
/* ── Branded Header Banner ── */
.app-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #1565C0 100%);
    padding: 1.8rem 2rem 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    color: white !important;
}
.app-header p { font-size: 0.95rem; opacity: 0.88; margin: 0; color: white !important; }
.app-header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    margin-right: 6px;
    margin-top: 10px;
    color: white;
}

/* --------------------------------------------------------
   PHASE SECTION HEADERS
   The blue left-bordered label at the start of each phase
   (e.g., "Phase 1 — Dataset Overview").
   -------------------------------------------------------- */
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #E3F2FD;
    border-left: 4px solid #1565C0;
    border-radius: 0 8px 8px 0;
    padding: 8px 16px;
    margin-bottom: 12px;
    font-weight: 600;
    font-size: 1.05rem;
    color: #0D47A1;
}

/* --------------------------------------------------------
   WORKFLOW STEPPER
   The horizontal 7-step progress bar. Three circle states:
     .step-done   → solid blue  (completed step)
     .step-active → outlined    (current step)
     .step-todo   → gray        (upcoming step)
   overflow-x: auto allows horizontal scrolling on small screens.
   -------------------------------------------------------- */
.stepper-wrap {
    display: flex;
    align-items: center;
    background: #F8F9FA;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 16px;
    overflow-x: auto;
    gap: 4px;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}
.step-circle {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    flex-shrink: 0;
}
.step-done  { background: #1565C0; color: white; }
.step-active{ background: #E3F2FD; color: #1565C0; border: 2px solid #1565C0; }
.step-todo  { background: #ECEFF1; color: #90A4AE; border: 2px solid #CFD8DC; }
.step-label { font-size: 12px; font-weight: 500; }
.step-label.done   { color: #1565C0; }
.step-label.active { color: #1565C0; font-weight: 700; }
.step-label.todo   { color: #90A4AE; }
.step-arrow { color: #B0BEC5; font-size: 14px; padding: 0 2px; }

/* --------------------------------------------------------
   METRIC CARDS
   Overrides Streamlit's default st.metric() styling with a
   light-blue card background and uppercase label formatting.
   data-testid selectors target Streamlit's internal component
   HTML structure (stable across recent Streamlit versions).
   -------------------------------------------------------- */
div[data-testid="metric-container"] {
    background: #F0F7FF;
    border: 1px solid #BBDEFB;
    border-radius: 10px;
    padding: 14px 16px !important;
}
div[data-testid="metric-container"] label {
    color: #1565C0 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0D47A1 !important;
}

/* --------------------------------------------------------
   PRIMARY BUTTON (Train All Models)
   Replaces Streamlit's default orange/red primary button with
   a blue gradient. The hover rule adds a subtle lift effect.
   -------------------------------------------------------- */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1565C0, #1976D2);
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    color: white;
    padding: 0.65rem 1.5rem;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(21,101,192,0.35);
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 4px 14px rgba(21,101,192,0.5);
    transform: translateY(-1px);
}

/* ── Section Dividers ── */
hr { border-color: #E3F2FD !important; }

/* --------------------------------------------------------
   METRICS DATAFRAME TEXT
   Forces white text inside the st.dataframe() metrics table.
   Without this, Streamlit's dark theme shows near-invisible
   light-gray text on the pandas Styler highlight cells.
   -------------------------------------------------------- */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* --------------------------------------------------------
   BEST MODEL ANALYSIS CARD
   The green card rendered after training that explains WHY
   one model is recommended for the specific business problem.
   -------------------------------------------------------- */
.analysis-card {
    background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
    border: 1px solid #A5D6A7;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 12px 0;
}
.analysis-card h4 { color: #1B5E20; margin: 0 0 10px 0; font-size: 1rem; }
.analysis-card p  { color: #2E7D32; margin: 0; font-size: 0.9rem; line-height: 1.6; }

/* --------------------------------------------------------
   WARNING/ERROR BOX
   Styled container for displaying warnings and errors.
   -------------------------------------------------------- */
.info-box {
    background: #FFF8E1;
    border-left: 4px solid #F9A825;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #E65100;
    margin: 8px 0;
}

/* --------------------------------------------------------
   SIDEBAR
   Blue gradient background applied to the entire left panel.
   `* { color: white }` makes all text inside the sidebar white
   so it stays readable against the dark background.
   The .stExpander rule gives sidebar expanders a frosted-glass look.
   -------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D47A1 0%, #1565C0 40%, #1976D2 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] h2, h3 { color: white !important; }
section[data-testid="stSidebar"] .stExpander {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
}
</style>
"""


def inject_css():
    """
    Push the custom CSS theme into the Streamlit page sir.
    Must be called once at the start of main() in app.py,
    before any other st.xxx() rendering calls sir.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
