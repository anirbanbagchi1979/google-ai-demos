import time as time
import numpy as np
from PIL import Image as PILImage
import streamlit as st
import backend

from vertexai.generative_models import (
    Part,
    Image,
)

st.set_page_config(
    layout="wide",
    page_title="Gaming Assets Assistant",
    page_icon="images/small-logo.png",
    initial_sidebar_state="expanded",
)


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def categorize(uploaded_images_list: list) -> None:
    with st.spinner("Generating Content..."):
        st.header("Gaming Assets Assistant")
        st.subheader("Automated Asset Categorization")
        i = 0
        asset_image_list = []
        for image_name in uploaded_images_list:
            image_name.save(f"images/model_image_{i}.png")
            asset_image = Part.from_image(
                Image.load_from_file(f"images/model_image_{i}.png")
            )
            asset_image_list.append(asset_image)
            i += 1

        st.write("Automatically categorize 3D assets based on multiple views")
        st.image(
            [
                uploaded_images_list[0],
                uploaded_images_list[1],
                uploaded_images_list[2],
                uploaded_images_list[3],
            ],
            width=100,
        )
        content = [
            """ You are an expert who looks at 3d models and provides details about the category of the assets.
                can you combine all these images and tell me what category this image belongs to. 
                ensure that the category is broad and it should be within 5 words. 
                give the reply in a json format as per the format below
                keep the details field within <15 words
        { "0. Model Category": ["Model Category content"], 
        "1. Details": ["Option", "Brief reasoning process"], """,
            "The images are here ",
            asset_image_list[0],
            asset_image_list[1],
            asset_image_list[2],
            asset_image_list[3],
            ".",
        ]

        tab1, tab2, tab3 = st.tabs(["Response", "Prompt","Timing"])
        with tab1:
            if content:
                with st.spinner("Generating Asset Info..."):
                    start_time = time.time()
                    response = backend.generate_image_classification(content)
                    end_time = time.time()
                    formatted_time = (
                        f"{end_time-start_time:.3f}"  # f-string for formatted output
                    )
                    st.json(response)
        with tab2:
            st.write("Prompt used:")
            st.write(content)
        with tab3:
            st.write("Time Taken:")
            st.write(formatted_time)


with st.sidebar:
    with st.form("Asset Classify"):
        uploaded_files = st.file_uploader(
            "Upload the images from different angles",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        image_classify_btn = st.form_submit_button("Categorize the Asset")
        if image_classify_btn:
            uploaded_images_list = []
            for uploaded_file in uploaded_files:
                # bytes_data = uploaded_file.read()
                st.write("filename:", uploaded_file.name)
                # st.write(bytes_data)
                image = PILImage.open(uploaded_file)
                img_array = np.array(image)
                # if image is not None:
                #     st.image(
                #         image,
                #         caption=f"You amazing image has shape {img_array.shape[0:2]}",
                #         use_column_width=True,
                #     )
                uploaded_images_list.append(image)

if image_classify_btn:
    categorize(uploaded_images_list)

st.logo("images/investments.png")
