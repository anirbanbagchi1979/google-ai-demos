import streamlit as st
import vertexai
from google.cloud import storage
import backend

favicon = "images/small-logo.png"

st.set_page_config(
    layout="wide",
    page_title="Gaming Asset Search",
    page_icon=favicon,
    initial_sidebar_state="expanded",
)
with st.spinner("Warming up all the services... "):
    backend.initialize_backend()
    
st.logo("images/top-logo-1.png")

st.header("Cymbal Games")
st.subheader("Game Developers AI Powered Assistant")
st.image("images/gamedev-assistant-logo.png")