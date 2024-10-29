import streamlit as st
import vertexai
from google.cloud import storage

favicon = "images/small-logo.png"

st.set_page_config(
    layout="wide",
    page_title="Gaming Asset Search",
    page_icon=favicon,
    initial_sidebar_state="expanded",
)

st.logo("images/top-logo-1.png")

st.header("Welcome")

st.image("images/gamedev-assistant-logo.png")