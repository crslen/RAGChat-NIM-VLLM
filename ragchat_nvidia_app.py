import streamlit as st
from rag import ChatCSV
import re
import tempfile
import os
import dotenv

#from dotenv import load_dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv()

app_name = os.getenv("APP_NAME")

st.set_page_config(page_title=app_name)
st.session_state["thinking_spinner"] = st.empty()

def load_index():
    load = st.session_state["assistant"].ingest("", True, "")
    if load:
        with st.chat_message("assistant", avatar="./images/ragechatbot.png"):
            st.write(load)

def clear_index():
    st.session_state["assistant"].check_kb(st.session_state["kb"])

def use_regex(input_text):
    x = re.findall(r"'http[^']*'", str(input_text))
    return x

def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.

    This function assumes that user input is stored in the Streamlit session state
    under the key "user_input," and the question-answering assistant is stored
    under the key "assistant."

    Additionally, it utilizes Streamlit functions for displaying a thinking spinner
    and updating the chat messages.

    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """

    # Check if there is user input and it is not empty.
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        # Extract and clean the user input.
        user_text = st.session_state["user_input"].strip()
        kb = st.session_state["kb"]
        prompt = st.session_state["prompt_input"]
        agent_text = st.session_state["assistant"].ask(user_text, kb, prompt)
        return agent_text

def read_and_save_url():
    # Clear the state of the question-answering assistant.
    st.session_state["assistant"].clear()

    #Ingest weblinks from session state
    st.session_state["assistant"].ingest(st.session_state["web_input"], False, "web")

def run_init():
    temp = st.session_state["temp"]
    dotenv.set_key(dotenv_file, "TEMPERATURE", str(temp))

def read_and_save_file():
    """
    Reads and saves the uploaded file, performs ingestion, and clears the assistant state.

    This function assumes that the question-answering assistant is stored in the Streamlit
    session state under the key "assistant," and file-related information is stored under
    the key "file_uploader."

    Additionally, it utilizes Streamlit functions for displaying spinners and updating the
    assistant's state.

    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Clear the state of the question-answering assistant.
    st.session_state["assistant"].clear()

    # Iterate through the uploaded files in the session state.
    for file in st.session_state["file_uploader"]:
        # Save the file to a temporary location and get the file path.
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Display a spinner while ingesting the file.
        file_ext = file.name
        if file_ext.endswith(".pdf"):
            st.session_state["assistant"].ingest(file_path, False, "pdf")
        elif file_ext.endswith(".csv"):
            st.session_state["assistant"].ingest(file_path, False, "csv")
        os.remove(file_path)

def page():
    """
    Defines the content of the Streamlit app page for ChatCSV.

    This function sets up the initial session state if it doesn't exist and displays
    the main components of the Streamlit app, including the header, file uploader,
    and associated functionalities.

    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    dotenv.load_dotenv(override=True)
    llm = os.getenv("LLM")
    temp = os.getenv("TEMPERATURE")

    default_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."""
    temp_help = """In most language models, the temperature range is typically between 0 and 1, where:
                    0: generates very predictable and similar text to the input (high confidence)
                    1: generates highly creative and novel text, but with a higher chance of errors (low confidence)
                For example, a temperature of 0.5 is a common default value that balances creativity and accuracy."""

    with st.sidebar:
        st.title(app_name)
        config = st.toggle("Show Configuration")
        if config:
            st.write(llm)  
            st.text_area("Prompt", default_prompt, key="prompt_input")
            col1, col2 = st.sidebar.columns(2)
            col1.button("Load Index",key="load_index", on_click=load_index)
            if col2.button("Clear chat history", key="clear_history", on_click=clear_index):
                st.session_state["assistant"].clear()
                st.session_state.trace_link = None
                st.session_state.run_id = None
            st.slider("Temperature", 0.0, 1.0, float(temp), 0.1, key="temp", help=temp_help, on_change=run_init)
        else:
            st.session_state["prompt_input"] = default_prompt
               
        st.text_area("Web Link(s)", key="web_input", on_change=read_and_save_url)
        st.file_uploader(
            "Upload PDF or CSV",
            type=["pdf","csv"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        st.checkbox("Use Knowledgebase", key="kb", value=False, on_change=clear_index)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state["assistant"] = ChatCSV()
        # load_index()
        st.session_state.messages = []
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            avatar = "./images/ragechatbot.png"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    # # User-provided prompt
    if prompt := st.chat_input("Ask me a question", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar='./images/ragechatbot.png'):
            with st.spinner("Thinking..."):
                response = process_input() 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

# Check if the script is being run as the main module.
if __name__ == "__main__":
    # Call the "page" function to set up and run the Streamlit app.
    page()