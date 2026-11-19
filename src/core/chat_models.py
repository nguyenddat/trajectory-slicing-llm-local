"""Local Ollama model shared by attribution and LLM-based slicing."""

from langchain_ollama import ChatOllama

from .settings import config

_ollama_url: str | None = None
_context_length: int | None = None
_models: dict[str, ChatOllama] = {}


def configure_ollama(url: str, context_length: int) -> None:
    global _ollama_url, _context_length
    _ollama_url = url
    _context_length = context_length
    _models.clear()


def get_chat_model(model_name: str) -> ChatOllama:
    if _ollama_url is None or _context_length is None:
        raise RuntimeError("Configure Ollama before creating methods")
    if model_name not in _models:
        _models[model_name] = ChatOllama(
            model=model_name,
            base_url=_ollama_url,
            temperature=config.model_temperature,
            seed=42,
            reasoning=False,
            num_ctx=_context_length,
            num_predict=256 if model_name.startswith("qwen3:") else None,
        )
    return _models[model_name]
