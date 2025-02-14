import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query, context, config):
    """
    Generate an answer from an LLM (OpenAI in this case) based on the user query and retrieved context segments.
    """
    client = OpenAI(api_key=api_key)

    model_name = config["openai"]["model_name"]
    temperature = config["openai"]["temperature"]

    system_prompt = (
        "Você é um assistente prestativo e especializado. "
        "Recebeu um contexto de um documento interno. "
        "Use o contexto fornecido para responder com precisão à pergunta do usuário. "
        "Se o contexto não for relevante, faça o seu melhor para fornecer uma resposta útil "
        "ou indique que não há informações disponíveis."
    )

    user_prompt = f"Contexto:\n{context}\n\nPergunta:\n{query}\n"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()
