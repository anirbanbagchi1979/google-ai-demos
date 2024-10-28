import streamlit as st
import pandas as pd
import numpy as np
from itables.streamlit import interactive_table
import pyarrow
from streamlit.components.v1 import html
from streamlit.components.v1.components import MarshallComponentException
from PIL import Image
from streamlit_navigation_bar import st_navbar
import pages as pg
# from css import *
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid
import time as time
import backend as data

favicon = 'images/small-logo.png'
st.set_page_config(
    layout="wide",
    page_title="Gaming Assets Assistant",
    page_icon=favicon,
    initial_sidebar_state="expanded",
)

def generate() ->None:
    with st.spinner("Generating Content..."):
        # start_time = time.time()
        text_story = data.generate_story(text_gen_prompt)
        # formatted_time = f"{time_spent:.3f}"  # f-string for formatted output
        # st.text(f"The Query took {formatted_time} seconds to complete.")
        # data_load_state = st.text('Loading data...')
        #   data_load_state.text('Loading data...done!')
        st.write(text_story)


with st.sidebar:
    with st.form("Asset Generator"):
        st.subheader("Prompt")
        text_gen_prompt = st.text_input("Input Prompt", value="Tell me a fancy story with a beauty and a beast within 200 words set in modern times")
        text_gen_prompt_submitted = st.form_submit_button("Tell Me a Story")
    if text_gen_prompt_submitted:
        time_to_generate = True;
    # st.write(generated_ouput)

if time_to_generate:
    generate()
st.logo("images/investments.png")




