import re
import unicodedata
from functools import cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
WHITESPACE_RE = re.compile(r"\s+")

@cache
def get_stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english"))

def normalize_string(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return WHITESPACE_RE.sub(" ", normalized).strip()

def remove_punctuation(text: str) -> str:
    return "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )

def remove_stop_words(words: list[str]) -> list[str]:
    stop_words = get_stop_words()
    return [word for word in words if word not in stop_words]

def stem_words(words: list[str]) -> list[str]:
    return [STEMMER.stem(word) for word in words]

def preprocess(content: str) -> list[str]:
    content = normalize_string(content)
    content = remove_punctuation(content)

    words = content.split()
    words = remove_stop_words(words)
    return stem_words(words)
