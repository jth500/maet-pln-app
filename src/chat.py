from utils import setup_logger
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv

# Config
load_dotenv()
logger = setup_logger(__name__)
GPT_PROMPT_TEMPLATE = """{input} TL;DR: """


@st.cache_resource(
    show_spinner="Loading the model. Don't worry, this only happens the first time!",
    ttl=24 * 3600,
)
def get_gpt_summarizer():
    return pipeline(
        "text-generation",
        model="ijwatson98/sft-gpt2-xsum-2703-tldr",
        tokenizer="gpt2-medium",
    )


@st.cache_data(show_spinner="Summarising with GPT2...")
def get_gpt_response(input):
    summarizer = get_gpt_summarizer()
    prompt = GPT_PROMPT_TEMPLATE.format(input=input)
    response = summarizer(prompt, truncation=True, max_length=10000)
    response = response[0]["generated_text"]
    i = response.index("TL;DR: ") + len("TL;DR: ")
    return response[i:]


def chat(input):
    logger.info("Chat function called")
    try:
        return get_gpt_response(input)
    except (IndexError, ValueError) as e:
        logger.exception(e)
        return "Oops! Something went wrong. Try again."
