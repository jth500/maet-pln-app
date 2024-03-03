import os
import logging
from pathlib import Path
import streamlit as st
from streamlit_chat import message
from streamlit_pills import pills

from chat import chat
from inputs import INPUTS
from utils import setup_logger

# Config
logger = setup_logger(__name__)
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


def write_user_response(r) -> None:
    with st.chat_message("user"):
        st.write(r)
    st.session_state.messages.append({"role": "user", "content": r})
    logger.debug(f"Printing user input: {r}")
    pass


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
    st.session_state.choice = None
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
        st.write(f"Model chosen: {st.session_state.model}")
        st.write(
            f"Info about {st.session_state.model}, who built it, number of parameters, etc."
        )

        pass


def button_callback_0():
    logger.debug("button_callback_1")
    st.session_state.choice = INPUTS[0]


def button_callback_1():
    logger.debug("button_callback_2")
    st.session_state.choice = INPUTS[1]


def button_callback_2():
    logger.debug("button_callback_3")
    st.session_state.choice = INPUTS[2]


def input_buttons():
    callbacks = [button_callback_0, button_callback_1, button_callback_2]
    cols = st.columns([1, 1, 1])
    st.write("Select a Reddit post from the dataset, or input your own below")
    for i, col in enumerate(cols):
        with col:
            st.button(INPUTS[i], key=f"button_{i}", on_click=callbacks[i])

    pass


def chat_flow():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if user_input := st.chat_input("Write your message here"):
        logger.info(f"User input received: {user_input}")
        st.session_state.choice = user_input
        st.rerun()

    elif st.session_state.choice:
        # Write the user response if there is one
        logger.info(f"No input received: {st.session_state.choice}")
        write_user_response(st.session_state.choice)
        pass

        response = chat(input=st.session_state.choice, model=st.session_state.model)
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        logger.info("No input received and no choice")


def main():
    st.title("maet-pln says hi")
    initialise_session_vars()
    sidebar()
    info_expander()
    chat_flow()
    input_buttons()
    pass


if __name__ == "__main__":
    main()
