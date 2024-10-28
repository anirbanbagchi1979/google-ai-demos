import streamlit as st
import pandas as pd
import numpy as np
from itables.streamlit import interactive_table
import pyarrow
from streamlit.components.v1 import html
from streamlit.components.v1.components import MarshallComponentException

from PIL import Image as PILImage
from streamlit_navigation_bar import st_navbar
import pages as pg

# from css import *
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid
import time as time
from google.cloud import storage

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Image,
)
import backend as model


favicon = "images/small-logo.png"
st.set_page_config(
    layout="wide",
    page_title="Gaming Assets Assistant",
    page_icon=favicon,
    initial_sidebar_state="expanded",
)


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def search(image_bytes,text_search) -> None:
    with st.spinner("Performing Image Search..."):
        st.header("Gaming Assets Assistant")
        st.subheader("Search for matching assets")
        image.save("images/model_image_to_search.png")
        # asset_image = Part.from_image(Image.load_from_file("images/model_house_incomplete.png"))
        asset_image = Part.from_image(Image.load_from_file("images/model_image_to_search.png"))
        st.image(
            image, width=100, caption="Image of the 3D Asset"
        )

        model.search(image_bytes,text_search)

with st.sidebar:
    with st.form("Asset Search"):
        img_search_file_buffer = st.file_uploader("Upload a matching photo", type=["png", "jpg", "jpeg"])
        text_search = st.text_input("What are you searching for?", value="A front view image of the model of a car")

        image_search_btn = st.form_submit_button("Search Matching Assets")
        if image_search_btn:
            image = PILImage.open(img_search_file_buffer)
            img_array = np.array(image)
            if image is not None:
                st.image(
                    image,
                    caption=f"You amazing image has shape {img_array.shape[0:2]}",
                    use_column_width=True,
                )
            image_file_contents =img_search_file_buffer.read()

if image_search_btn:
    search(image_file_contents,text_search)

st.logo("images/investments.png")
