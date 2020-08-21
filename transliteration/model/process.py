import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Tuple
from transliteration.model import START_TOKEN, END_TOKEN


def tokenize(
    lang: List[str],
) -> Tuple[np.ndarray, tf.keras.preprocessing.text.Tokenizer]:
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
    return tensor, lang_tokenizer


def create_dataset(
    path: Path, num_examples: int = None, start_token=START_TOKEN, end_token=END_TOKEN
):
    lines = path.open(encoding="utf-8").read().strip().split("\n")
    word_pairs = [
        [start_token + w + end_token for w in l.split(",")]
        for l in lines[:num_examples]
    ]
    return zip(*word_pairs)


def load_dataset(path, num_examples=None):
    # creating cleaned source, target pairs
    source, target = create_dataset(path, num_examples)

    source_tensor, source_tokenizer = tokenize(source)
    target_tensor, target_tokenizer = tokenize(target)

    return source_tensor, target_tensor, source_tokenizer, target_tokenizer
