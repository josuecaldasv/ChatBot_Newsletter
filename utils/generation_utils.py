# generate_utils.py
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query, context, config):
    """
    Generate an answer from an LLM (OpenAI) based on the user query and retrieved context segments.
    """
    client = OpenAI(api_key=api_key)

    model_name = config["openai"]["model_name"]
    temperature = config["openai"]["temperature"]

    system_prompt = (
        "You are a helpful and knowledgeable assistant. "
        "You have been given some context from an internal document. "
        "Please use the provided context to answer the user's question accurately. "
        "If the context is not relevant, do your best to provide a helpful answer or "
        "indicate no information is available."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()