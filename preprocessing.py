

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pymorphy3

# Инициализация инструментов
stemmer_en = SnowballStemmer("english")
morph = pymorphy3.MorphAnalyzer()

# Стоп-слова для обоих языков
STOP_WORDS = set(stopwords.words("russian")) | set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^а-яёa-z\s]", " ", text)  # оставляем только буквы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_russian(word: str) -> bool:
    return bool(re.search(r"[а-яё]", word))


def tokenize(text: str) -> list[str]:
    text = clean_text(text)
    tokens = word_tokenize(text, language="russian")
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return tokens


def process_token(token: str) -> str:
    if is_russian(token):
        return morph.parse(token)[0].normal_form
    else:
        return stemmer_en.stem(token)


def preprocess(text: str, method: str = "auto") -> list[str]:
    tokens = tokenize(text)
    return [process_token(t) for t in tokens]