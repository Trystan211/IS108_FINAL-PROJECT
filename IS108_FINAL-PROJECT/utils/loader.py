"""
utils/loader.py
Handles reading the uploaded CSV or Excel file into a Pandas DataFrame.
This is kept in its own file so the loading logic stays separate from
preprocessing and UI code sir.
"""
import pandas as pd
import streamlit as st


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a DataFrame.
 
    @st.cache_data caches the result using the file's content as the key.
    This means re-running the app (e.g., user moves a slider) does NOT
    re-read the file from disk — it returns the already-parsed DataFrame
    instantly, keeping the app fast.
 
    Returns an empty DataFrame on failure sir (error is shown to the user).
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported format. Please upload .csv or .xlsx.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return pd.DataFrame()