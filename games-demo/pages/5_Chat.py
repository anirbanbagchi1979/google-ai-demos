import streamlit as st
import backend

st.title(f"Chatbot with Gemini-Flash-1.5 🤖")

st.markdown(
    "To read more about Gemini Flash 1.5, a cutting-edge language model from Google AI."
    "check https://deepmind.google/technologies/gemini/flash/"
)

client = backend.get_chat_client()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = backgemini_15_flash._model_name

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})