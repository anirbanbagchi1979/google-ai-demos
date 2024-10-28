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


favicon = 'images/small-logo.png'

st.set_page_config(
    layout="wide",
    page_title="FinVest Advisor",
    page_icon=favicon,
    initial_sidebar_state="expanded",
)

with st.sidebar:

    with st.form("Asset Search"):
        st.subheader("Search Criteria")
        preciseVsText = st.radio("", ["Full-Text", "Precise"], horizontal=True)
        preciseSearch = False
        with st.expander("Asset Strategy", expanded=True):
            investment_strategy_pt1 = st.text_input("", value="Europe")
            andOrExclude = st.radio("", ["AND", "OR", "EXCLUDE"], horizontal=True)
            investment_strategy_pt2 = st.text_input("", value="Asia")
        investment_manager = st.text_input("Investment Manager", value="James")
        if preciseVsText == "Full-Text":
            if andOrExclude == "EXCLUDE":
                investment_strategy = (
                    investment_strategy_pt1 + " -" + investment_strategy_pt2
                )
            else:
                investment_strategy = (
                    investment_strategy_pt1
                    + " "
                    + andOrExclude
                    + " "
                    + investment_strategy_pt2
                )
        else:
            preciseSearch = True
        asset_search_submitted = st.form_submit_button("Submit")
if asset_search_submitted:
    if preciseSearch:
        # asset_search_precise()
        st.write("asset search")
    else:
        # asset_search()
        st.write("asset search else")

st.logo("images/investments.png")