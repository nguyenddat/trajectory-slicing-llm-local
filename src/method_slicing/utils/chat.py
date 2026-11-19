import time

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser

def get_chat_completion(
        model: any,
        prompt_template: str,
        parser,
        prompt_params: BaseModel,
) -> tuple[dict, dict]:

    prompt_messages = ChatPromptTemplate.from_messages(
        [("human", prompt_template + "\n{format_instructions}")]
    ).partial(format_instructions=parser.get_format_instructions())

    prompt_value = prompt_messages.invoke(prompt_params.model_dump())

    t0 = time.perf_counter()
    ai_msg = model.invoke(prompt_value)
    latency = time.perf_counter() - t0

    cached_tokens = (ai_msg.usage_metadata.get("input_token_details") or {}).get("cache_read")
    if cached_tokens is None:
        token_usage = ai_msg.response_metadata.get("token_usage") or {}
        cached_tokens = (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    cost = {
        "latency": latency,
        "input_tokens": ai_msg.usage_metadata["input_tokens"],
        "output_tokens": ai_msg.usage_metadata["output_tokens"],
        "cached_tokens": cached_tokens,
    }

    try:
        result = parser.invoke(ai_msg).model_dump()
    except Exception:  # noqa: BLE001 - fall back to output-fixing on any parse failure
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
        result = fixing_parser.invoke(ai_msg).model_dump()
    return result, cost
