import time as time
from collections import OrderedDict
import streamlit as st
import numpy as np
from PIL import Image as PILImage
import backend
import requests

def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def search(image_bytes, text_search, pil_image, search_by_image) -> None:
    with st.spinner("Performing Image Search..."):
        st.subheader("AI Powered Asset Search")
        if pil_image != "":
            st.image(pil_image, width=100, caption="Image of the 3D Asset")
        st.subheader("Matched Assets Loading...")

        image_uris = backend.search_image_warehouse(
            image_bytes, text_search, search_by_image
        )
        # print(f"Image Fetching from {image_uris[0]}")
        with st.spinner("Loading..."):
            with requests.Session() as s:
                headers = OrderedDict(
                    [
                        ("Accept", "application/json, text/plain, */*"),
                        ("Accept-Encoding", "identity"),
                        ("Connection", "close"),
                        ("Referer", "https://www.oui.sncf/bons-plans/tgvmax"),
                        (
                            "User-Agent",
                            "Mozilla/5.0 (X11; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0",
                        ),
                        ("X-CT-Locale", "fr"),
                        ("X-User-Agent", "CaptainTrain/1542980776(web) (Ember 3.4.6)"),
                        ("content-type", "application/json"),
                        ("x-not-a-bot", "i-am-human"),
                    ]
                )
                # Create columns for the images
                cols = st.columns(len(image_uris))
                response_contents = []
                # Iterate over the image URIs and display images in each column
                for i, uri in enumerate(image_uris):
                    with cols[i]:  # Use the context manager for each column
                        response = s.get(uri, timeout=1, headers=headers)
                        st.image(response.content, width=100)

with st.sidebar:
    with st.form("Asset Search"):
        image_or_text_search = st.radio(
            "", ["Search By Image", "Text"], horizontal=True
        )
        uploaded_file = st.file_uploader(
            "Upload photo to search", type=["png", "jpg", "jpeg"]
        )
        text_search = st.text_input("What are you searching for?", value="An aeroplane")

        image_search_btn = st.form_submit_button("Search Matching Assets")
        if image_search_btn:
            pil_image = ""
            if image_or_text_search == "Search By Image":
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
    search_by_image = False
    if image_or_text_search == "Search By Image":
        search_by_image = True

    search(uploaded_file, text_search, pil_image, search_by_image)

st.logo("images/top-logo-1.png")
