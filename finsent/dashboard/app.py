import streamlit as st


def main() -> None:
    """Render the Streamlit dashboard scaffold."""
    st.set_page_config(page_title="Financial Sentiment Lab", layout="wide")
    st.title("Financial Sentiment Lab")
    st.info("Dashboard scaffold. Pipeline data is not connected yet.")
