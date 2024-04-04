from pathlib import Path
import streamlit as st
import pandas as pd

from chat import chat
from utils import setup_logger
from text_elements import sidebar, info_expander
import time

# Config
logger = setup_logger(__name__)
logger.debug("Running from top")  # just useful to undserstand the order of execution
DATA = Path(__file__).parents[1] / "data"
st.set_page_config(
    page_title="maet pln",
)


def initialise_session_vars():
    if "messages" not in st.session_state:
        logger.debug("Initialising messages")
        st.session_state.messages = []
    if "choice" not in st.session_state:
        logger.debug("Initialising choice")
        st.session_state.choice = None
    if "sft_summaries" not in st.session_state:
        st.session_state.sft_summaries = []
    if "sft_sum_choice" not in st.session_state:
        st.session_state.sft_sum_choice = None
    if "summary_msgs" not in st.session_state:
        st.session_state.summary_msgs = []


@st.cache_data(show_spinner=True)
def get_summaries():
    return pd.read_csv(DATA / "gpt_summaries.csv", index_col=0)


def get_random_summaries(seed=None):
    df = get_summaries()
    row = df.sample(n=1)
    return row.iloc[0, [0, 1, 2, 4]]


def random_summary_callback():
    sum = get_random_summaries()
    st.session_state.sft_sum_choice = sum.to_dict()


def article_selection_buttons():
    s = "Select a random news article from the dataset, or input your own text below."
    st.write(s)
    st.button("Select a random article.", on_click=random_article_callback)


@st.cache_data(show_spinner=True)
def get_example_articles():
    return pd.read_csv(DATA / "sample_articles.csv", index_col=0)


def get_random_article(seed=None):
    df = get_example_articles()
    row = df.sample(n=1)
    return row.iloc[0, 0]


def random_article_callback():
    article = get_random_article()
    st.session_state.choice = article


def summary_selection_button():
    st.write("Select a set of summaries from a pre-populated dataset.")
    st.button("Select a random article.", on_click=random_summary_callback, key=5)


def write_user_msg(r) -> None:
    with st.chat_message("user"):
        st.write(r)


def write_multi_summary(responses):
    cols = st.columns([1, 1, 1])
    names = ["GPT", "T5", "True Summary"]
    for col, r, name in zip(cols, responses.values(), names):
        with col:
            st.write(f"**{name}**")
            st.markdown(r)


def sft_chat_flow():
    st.session_state.choice = None
    for m in st.session_state.summary_msgs:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"])
            else:
                write_multi_summary(m["content"])

    if responses := st.session_state.sft_sum_choice:
        article = responses["input"]
        write_user_msg(article)
        with st.spinner("One sec..."):
            time.sleep(1)
        # Add the article as a user message, then the summaries as a response
        st.session_state.summary_msgs.append({"role": "user", "content": article})
        st.session_state.summary_msgs.append(
            {"role": "assistant", "content": responses}
        )
        with st.chat_message("assistant"):
            write_multi_summary(responses)

    else:
        logger.info("No input received and no choice")


def chat_flow(user_input):
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if user_input:
        logger.info(f"User input received: {user_input}")
        st.session_state.choice = user_input
        st.session_state.choice_last = user_input
        st.rerun()

    elif st.session_state.choice:
        logger.info(f"No input received: {st.session_state.choice}")
        write_user_msg(st.session_state.choice)
        st.session_state.messages.append(
            {"role": "user", "content": st.session_state.choice}
        )
        pass

        response = chat(input=st.session_state.choice)
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

    user_input = st.chat_input("Write your message here")
    tab1, tab2 = st.tabs(
        ["🤖 Reinforcement Learning with AI Feedback", ":chart: Supervised Fine Tuning"]
    )

    with tab1:
        chat_flow(user_input)
        article_selection_buttons()
    with tab2:
        sft_chat_flow()
        summary_selection_button()


if __name__ == "__main__":
    main()
