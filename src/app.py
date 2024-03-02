import os
import logging
from pathlib import Path
import streamlit as st
from streamlit_chat import message
from streamlit_pills import pills

from chat import chat
from inputs import INPUTS

# Config
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
cwd = Path(os.getcwd())

logger.debug("Running from top")  # just useful to undserstand the order of execution


def initialise_session_vars():
    if "messages" not in st.session_state:
        logger.debug("Initialising messages")
        st.session_state.messages = []
    if "choice" not in st.session_state:
        logger.debug("Initialising choice")
        st.session_state.choice = None
    if "model" not in st.session_state:
        logger.debug("Initialising model")
        st.session_state.model = "T5"


def radio_callback():
    logger.debug("radio_callback")
    st.session_state.choice = st.session_state.choice_radio
    pass


def write_user_response(r) -> None:
    with st.chat_message("user"):
        st.write(r)
    st.session_state.messages.append({"role": "user", "content": r})
    logger.debug(f"Printing user input: {r}")
    pass


def input_form():
    with st.form("form"):
        prompt = "Select a Reddit post from the dataset, or input your own below"
        choice = st.radio(prompt, options=INPUTS, key="choice_radio", index=None)
        submit = st.form_submit_button("Submit", on_click=radio_callback)


def info_expander():
    with st.expander("Built by Group 1, 2, 3, 4, 5 for COMP0087"):
        st.write(
            "This project is for COMP0087. We are 1, 2, 3, 4, 5. We do XX. We use YY. We are ZZZ.\n"
            "We are doing this project because we want to learn about AA, BB, CC.\n"
        )
        st.write(
            "We are using this dataset because we want to learn about DD, EE, FF.\n"
            "We are using this model because we want to learn about GG, HH, II. "
        )


def model_callback():
    logger.debug(st.session_state.model)
    pass


def sidebar():
    # TBD What should the radio do - presumably it should change
    with st.sidebar:
        st.title("More info about the project")
        st.write("This is the sidebar. info about the project, etc. add some more text")
        st.write("This is the sidebar. info about the project, etc. add some more text")
        st.write("This is the sidebar. info about the project, etc. add some more text")
        st.radio(
            "Select a model",
            ("T5", "GPT", "BART"),
            on_change=model_callback,
            key="model",
        )

        pass


def chat_flow():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if user_input := st.chat_input("Write your message here"):
        logger.info(f"User input received: {user_input}")
        st.session_state.choice = user_input
        st.rerun()

    else:
        # Write the user response if there is one
        logger.info(f"No input received: {st.session_state.choice}")
        if st.session_state.choice:
            write_user_response(st.session_state.choice)
            pass

        response = chat(st.session_state.choice)
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.title("maet-pln says hi")
    info_expander()
    chat_flow()
    input_form()
    pass


if __name__ == "__main__":
    initialise_session_vars()
    sidebar()
    main()
