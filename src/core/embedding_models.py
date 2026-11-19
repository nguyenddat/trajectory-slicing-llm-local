from langchain_openai import OpenAIEmbeddings

from .settings import config

model_names = ["openai/text-embedding-3-small"]
model_cache = {}

def get_embedding_model(model_name: str) -> OpenAIEmbeddings:
    if model_name not in model_cache:
        if model_name not in model_names:
            raise ValueError(f"Model '{model_name}' not found. Available models: {model_names}")

        model_cache[model_name] = OpenAIEmbeddings(
            model=model_name,
            api_key=config.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return model_cache[model_name]
