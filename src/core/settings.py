"""Settings required by the local runner and optional transformer slicer."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    model_temperature: int = 0
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")


config = Config()
