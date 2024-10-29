import time as time
import numpy as np
import streamlit as st
from PIL import Image as PILImage
from vertexai.generative_models import (
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


def quality_check(image) -> None:
    with st.spinner("Generating Content..."):
        st.header("Gaming Assets Assistant")
        st.subheader("Generate Automated Asset Quality Report")
        image.save("images/model_image.png")
        # asset_image = Part.from_image(Image.load_from_file("images/model_house_incomplete.png"))
        asset_image = Part.from_image(Image.load_from_file("images/model_image.png"))
        st.image(
            image, width=100, caption="Image of the 3D Asset"
        )
        st.write("Automatically check quality of 3D assets in multiple categories")
        content = [
            """ You are an expert who looks at 3d models and provides details about the quality of the assets.
        You need to provide the following details in a JSON format by looking at the image  which is attached to this prompt.
        Example of the JSON output is as below. Also provide the brief reasoning to be within 100 words
        { "0. Model Description": ["Model description content"], 
        "1. Belongs to Category": ["Option", "Brief reasoning process"], 
        "2. Incomplete Model": ["Option", "Brief reasoning process"], 
        "3. Quantity exceeds 1": ["Option", "Brief reasoning process"], 
        "4. 2D model in 3D space": ["Option", "Brief reasoning process"], 
        "5. Simple geometric shape/structure": ["Option", "Brief reasoning process"], 
        "6. Simple texture details": ["Option", "Brief reasoning process"], 
        "7. Lacks bottom face": ["Option", "Brief reasoning process"], 
        "8. Has platform at the bottom": ["Option", "Brief reasoning process"], 
        "9. Surface damage": ["Option", "Brief reasoning process"], 
        "10. Surface aging": ["Option", "Brief reasoning process"] }  """,
            "The image is here ",
            asset_image,
            ".",
        ]

        tab1, tab2, tab3 = st.tabs(["Response", "Prompt", "Timing"])
        with tab1:
            if content:
                with st.spinner("Generating Asset Info..."):
                    start_time = time.time()
                    response = model.generate_image_classification(content)
                    end_time = time.time()
                    formatted_time = f"{end_time-start_time:.3f} seconds"  # f-string for formatted output

                    st.code(response)

        with tab2:
            st.write("Prompt used:")
            st.write(content)
        with tab3:
            st.write("Time Taken:")
            st.write(formatted_time)

with st.sidebar:
    with st.form("Asset Classify"):
        img_file_buffer = st.file_uploader("Upload an Asset", type=["png", "jpg", "jpeg"])
        image_classify_btn = st.form_submit_button("Check Quality")
        if image_classify_btn:
            image = PILImage.open(img_file_buffer)
            img_array = np.array(image)
            if image is not None:
                st.image(
                    image,
                    caption=f"You amazing image has shape {img_array.shape[0:2]}",
                    use_column_width=True,
                )

if image_classify_btn:
    quality_check(image)

st.logo("images/investments.png")
