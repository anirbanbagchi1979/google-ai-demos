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
import requests
from io import BytesIO
from collections import OrderedDict

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


def search(image_bytes,text_search,pil_image) -> None:
    with st.spinner("Performing Image Search..."):
        st.header("Gaming Assets Assistant")
        st.subheader("Search for matching assets")
        # image.save("images/model_image_to_search.png")
        # asset_image = Part.from_image(Image.load_from_file("images/model_house_incomplete.png"))
        # asset_image = Part.from_image(Image.load_from_file("images/model_image_to_search.png"))
        st.image(
            pil_image, width=100, caption="Image of the 3D Asset"
        )
        st.subheader("Matched Assets Loading...")

        image_uris = model.search_image_warehouse(image_bytes,text_search)
        # index = st.slider("Select an image", 0, len(image_uris) - 1, 0)  # Slider to select image
        # Fetch the image using requests.get
        print(f"Image Fetching from {image_uris[0]}")
        with st.spinner("Loading..."):
            with requests.Session() as s:
                headers = OrderedDict([('Accept', 'application/json, text/plain, */*'),
                        ('Accept-Encoding', 'identity'),
                        ('Connection', 'close'),
                        ('Referer', 'https://www.oui.sncf/bons-plans/tgvmax'),
                        ('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0'),
                        ('X-CT-Locale', 'fr'),
                        ('X-User-Agent', 'CaptainTrain/1542980776(web) (Ember 3.4.6)'),
                        ('content-type', 'application/json'),
                        ('x-not-a-bot', 'i-am-human')])
                        # Create columns for the images
                cols = st.columns(len(image_uris))  
                response_contents = []
                # Iterate over the image URIs and display images in each column
                for i, uri in enumerate(image_uris):
                    with cols[i]:  # Use the context manager for each column
                        response = s.get(uri, timeout=1, headers=headers)
                        st.image(response.content, width=100)
                        # response_contents.append(response.content)
                        # print(f"Image fetched from {image_uris[0]}")
            # st.image(response_contents,width=100)
        # BytesIO.raise_for_status()  # Raise an exception for bad status codes

        # Display the image using st.image
        
        # print(f"number of respinse images {len(response_images)}")
        # st.image(response_images)

with st.sidebar:
    with st.form("Asset Search"):
        uploaded_file = st.file_uploader("Upload photo to search", type=["png", "jpg", "jpeg"])
        text_search = st.text_input("What are you searching for?", value="An aeroplane")

        image_search_btn = st.form_submit_button("Search Matching Assets")
        if image_search_btn:
            pil_image = PILImage.open(uploaded_file)
            img_array = np.array(pil_image)
            if pil_image is not None:
                st.image(
                    pil_image,
                    caption=f"You amazing image has shape {img_array.shape[0:2]}",
                    use_column_width=True,
                )
            # image_file_contents =img_search_file_buffer.read()

if image_search_btn:
    search(uploaded_file,text_search,pil_image)

st.logo("images/investments.png")
