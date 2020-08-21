import json
from pathlib import Path

from tensorflow.python.keras.preprocessing.text import tokenizer_from_json


def save_keras_tokenizer_json(tokenizer, path: Path):
    # save Keras tokenizer for input language (arabic)
    tokenizer_json = tokenizer.to_json()
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def load_keras_tokenizer_json(path: Path):
    # load Keras tokenizer for input language (arabic)
    with path.open() as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer


def save_model_metadata(metadata, path: Path):
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False))


def load_model_metadata(path: Path):
    # load Keras tokenizer for input language (arabic)
    with path.open() as f:
        data = json.load(f)
    return data
