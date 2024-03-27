import os
from utils import setup_logger
from pathlib import Path
import streamlit as st
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# from trl.core import LengthSampler
from dotenv import load_dotenv


# Config
logger = setup_logger(__name__)
load_dotenv()


def generate_t5_prompt(input):
    return f"summarize: {input}"


def generate_gpt_prompt(input):
    return f"""You are an expert in text summarization. You are given the full text. Your job is to summarise the text as concisely and accurately as possible.

    ### Input:
    {input}"""


@st.cache_resource(
    show_spinner="Loading the GPT-2 model. Don't worry, this only happens the first time you run GPT2!"
)
def get_gpt_summarizer():
    return pipeline(
        "text-generation",
        model="ijwatson98/sft-gpt2-xsum-2503",
        tokenizer="gpt2-medium",
    )


@st.cache_resource(show_spinner="Summarising with GPT2...")
def get_gpt_response(input):
    summarizer = get_gpt_summarizer()
    prompt = generate_gpt_prompt(input)
    response = summarizer(prompt, truncation=True, max_length=10000)
    response = response[0]["generated_text"]
    i = response.index("SUMMARY:") + len("SUMMARY:\n")
    return response[i:]


@st.cache_resource(
    show_spinner="Loading the T5 model. Don't worry, this only happens the first time you run T5!"
)
def get_t5_summarizer():
    return pipeline("summarization", model="jth500/t5-base-v3.1", tokenizer="t5-base")


@st.cache_resource(show_spinner="Summarising with T5...")
def get_t5_response(input):
    summarizer = get_t5_summarizer()
    prompt = generate_t5_prompt(input)
    response = summarizer(prompt)[0]["summary_text"]
    return response


def default_response(input):
    return f"This is a placeholder for the chat function. Input: {input}"


def chat(input, model, *args, **kwargs):
    logger.info("Chat function called")
    response_funcs = {"GPT": get_gpt_response, "T5": get_t5_response}
    try:
        get_response = response_funcs.get(model, default_response)
        return get_response(input)
    except (IndexError, ValueError) as e:
        logger.exception(e)
        return "Oops! Something went wrong. Try again."
