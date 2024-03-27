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
cwd = Path(os.getcwd())
load_dotenv()


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "ijwatson98/sft-gpt2-xsum-1503"
    )
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
    return tokenizer, model


def generate_t5_prompt(input):
    return f"summarize: {input}"


def generate_prompt(input, output=""):
    return f"""You are an expert in text summarization. You are given the full text. Your job is to summarise the text as concisely and accurately as possible.

    ### Input:
    {input}

    ### Response:
    {output}"""


def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        # truncation=True,
        # max_length=512,
        # padding=False,
        # return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(
        data_point,
    )
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    return tokenized_full_prompt


def do_it(model, prompt_tensor, tokenizer):
    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "temperature": 0.5,
        "min_length": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
    }

    # prompt_tensor = val_data[0]['input_ids']
    # prompt_tensor = torch.tensor(prompt_tensor).unsqueeze(dim=0).to(device)
    max_new_tokens = output_length_sampler()
    generation_kwargs["max_new_tokens"] = max_new_tokens
    summary_tensor = model.generate(input_ids=prompt_tensor, **generation_kwargs)
    summary = tokenizer.decode(summary_tensor[0], skip_special_tokens=True)
    return summary


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


def chat(input, model, *args, **kwargs):
    logger.info("Chat function called")
    if model == "GPT":
        # model, tokenizer = get_model()
        # prompt_tensor = generate_and_tokenize_prompt(input, tokenizer)["input_ids"]
        # response = do_it(model, prompt_tensor, tokenizer)
        # r_ind = response.index("Response:\n") + len("Response:\n")
        # return response[r_ind:]
        pass
    elif model == "T5":
        return get_t5_response(input)

    return f"This is a placeholder for the chat function. Model chosen: {model}. Input: {input}"
