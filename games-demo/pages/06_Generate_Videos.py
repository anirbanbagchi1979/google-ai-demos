import time as time
import numpy as np
import streamlit as st
from PIL import Image as PILImage
from vertexai.generative_models import (
    Part,
    Image,
)
import json
import backend as model


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def generate_videos(image, original_prompt) -> None:
    with st.spinner("Generating Content..."):
        st.subheader("AI Powered Image to Video Generator")
        # if imaimage.save("images/model_image.png")
        # asset_image = Part.from_image(Image.load_from_file("images/model_house_incomplete.png"))
        # asset_image = Part.from_image(Image.load_from_file("images/model_image.png"))
        # st.image(
        #     image, width=100, caption="Image of the 3D Asset"
        # )
        # st.write("Automatically check quality of 3D assets in multiple categories")
        text_to_video_prompt_guide = """
            Text-to-Video Prompt Writing Help:
            <text-to-video-prompt-guide>
            Here are some our best practices for text-to-video prompts:

            Detailed prompts = better videos:
            - More details you add, the more control you’ll have over the video.
            - A prompt should look like this: "Camera dollies to show a close up of a desperate man in a green trench coat is making a call on a rotary style wall-phone, green neon light, movie scene."
                - Here is a break down of elements need to create a text-to-video prompt using the above prompt as an example:
                - "Camera dollies to show" = "Camera Motion"
                - "A close up of" = "Composition"
                - "A desperate man in a green trench coat" = "Subject"
                - "Is making a call" = "Action"
                - "On a roary style wall-phone" = "Scene"
                - "Green Neon light" = "Ambiance"
                - "Movie Scene" = "Style"

            Use the right keywords for better control:
            - Here is a list of some keywords that work well with text-to-video, use these in your prompts to get the desired camera motion or style.
            - Subject: Who or what is the main focus of the shot.  Example: "happy woman in her 30s".
            - Scene: Where is the location of the shot. Example "on a busy street, in space".
            - Action: What is the subject doing Examples: "walking", "running", "turning head".
            - Camera Motion: What the camera is doing. Example: "POV shot", "Aerial View", "Tracking Drone view", "Tracking Shot".

            Example text-to-video prompt using the above keywords:
            - Example text-to-video prompt: "Tracking drone view of a man driving a red convertible car in Palm Springs, 1970s, warm sunlight, long shadows"
            - Example text-to-video prompt: "A POV shot from a vintage car driving in the rain, Canada at night, cinematic"

            Styles:
            - Overall aesthetic. Consider using specific film style keywords.  Examples: "horror film", "film noir, "animated styles", "3D cartoon style render".
            - Example text-to-video prompt: "Over the shoulder of a young woman in a car, 1970s, film grain, horror film, cinematic he Film noir style, man and woman walk on the street, mystery, cinematic, black and white"
            - Example text-to-video prompt: "A cute creatures with snow leopard-like fur is walking in winter forest, 3D cartoon style render. An architectural rendering of a white concrete apartment building with flowing organic shapes, seamlessly blending with lush greenery and futuristic elements."

            Composition:
            - How the shot is framed. This is often relative to the subject e.g. wide shot, close-up, low angle
            - Example text-to-video prompt: "Extreme close-up of a an eye with city reflected in it. A wide shot of surfer walking on a beach with a surfboard, beautiful sunset, cinematic"

            Ambiance & Emotions:
            - How the color and light contribute to the scene (blue tones, night)
            - Example text-to-video prompt: "A close-up of a girl holding adorable golden retriever puppy in the park, sunlight Cinematic close-up shot of a sad woman riding a bus in the rain, cool blue tones, sad mood"

            Cinematic effects:
            - e.g. double exposure, projected, glitch camera effect.
            - Example text-to-video prompt: "A double exposure of silhouetted profile of a woman walking and lake, walking in a forest Close-up shot of a model with blue light with geometric shapes projected on her face"
            - Example text-to-video prompt: "Silhouette of a man walking in collage of cityscapes Glitch camera effect, close up of woman’s face speaking, neon colors"
            </text-to-video-prompt-guide>
            """

        prompt_to_gemini = f"""Rewrite the following "original prompt" using the text-to-video instructions below.
            You want the video to be creative and artistic..

            Original Prompt:
            "{original_prompt}"

            Output Fields:
            - "text-to-video-prompt":

            Instructions:
            - Read the  "Text-to-Video Prompt Writing Help" to learn more about how to create good text-to-video prompts.
            - Make sure you include all the relevant best practices when creating the text-to-video prompt.
            - Do not include "text overlays" in the text-to-video prompt.
            - Do not include children in the text-to-video prompt.

            {text_to_video_prompt_guide}
            """
        response_schema = {
            "type": "object",
            "required": ["text-to-video-prompt"],
            "properties": {"text-to-video-prompt": {"type": "string"}},
        }
        # print(f"Prompt to gemini : {prompt_to_gemini}")
        st.image(image, width=200)
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Response", "Prompt", "Timing", "Video Generate"]
        )
        with tab1:
            if prompt_to_gemini:
                with st.spinner("Generating Better Prompt..."):
                    start_time = time.time()
                    response = model.generate_story(
                        prompt_to_gemini, response_schema=response_schema
                    )
                    end_time = time.time()
                    formatted_time = f"{end_time-start_time:.3f} seconds"  # f-string for formatted output
                    st.write(response)
                    refined_prompt_dict = json.loads(response)
                    # print(refined_prompt_dict)
                with st.spinner("Generating Video ..."):
                    filename = model.generate_video(
                        image, refined_prompt_dict[" text-to-video-prompt"]
                    )
                    st.write(f"Video File Created  : {filename}")

        with tab2:
            st.write("Prompt used:")
            st.write(prompt_to_gemini)
        with tab3:
            st.write("Time Taken:")
            st.write(formatted_time)


with st.sidebar:
    with st.form("Generate Video"):
        img_file_buffer = st.file_uploader("Upload a starting image", type=["png", "jpg", "jpeg"])
        input_prompt = st.text_area(
            "Describe what you want from the video?",
            value="Make a 3d spinning video out of this input image. ",
        )

        generate_video_btn = st.form_submit_button("Generate Video")
        if generate_video_btn:
            image = PILImage.open(img_file_buffer)
        # img_array = np.array(image)
        # if image is not None:
        #     st.image(
        #         image,
        #         caption=f"You amazing image has shape {img_array.shape[0:2]}",
        #         use_column_width=True,
        #     )

if generate_video_btn:
    generate_videos(image, input_prompt)

st.logo("images/top-logo-1.png")
