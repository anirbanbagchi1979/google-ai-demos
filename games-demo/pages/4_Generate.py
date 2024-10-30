import streamlit as st
import time as time
import backend

# st.set_page_config(
#     layout="wide",
#     page_title="Gaming Assets Assistant",
#     # page_icon=favicon,
#     initial_sidebar_state="expanded",
# )


def generate() -> None:
    with st.spinner("Generating Content..."):
        start_time = time.time()
        text_story = backend.generate_story(text_gen_prompt)
        end_time = time.time()
        formatted_time = f"{end_time-start_time:.3f}"  # f-string for formatted output
        
        st.subheader("AI Powered Story Generator")
        with st.expander("Timing"):
                # with stylable_container(
                #     "codeblock",
                #     """
                # code {
                #     white-space: pre-wrap !important;
                # }
                # """,
                # ):
                st.text(f"The Query took {formatted_time} seconds to complete.")
        st.write(text_story)


with st.sidebar:
    time_to_generate = False
    with st.form("Asset Generator"):
        st.subheader("Prompt")
        text_gen_prompt = st.text_area(
            "Input Prompt",
            value="Tell me a fancy story with a beauty and a beast within 200 words set in modern times",
        )
        text_gen_prompt_submitted = st.form_submit_button("Tell Me a Story")
    if text_gen_prompt_submitted:
        time_to_generate = True
    # st.write(generated_ouput)

if time_to_generate:
    generate()
st.logo("images/top-logo-1.png")
