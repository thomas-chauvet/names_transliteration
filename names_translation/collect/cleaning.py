import re
import string
from pathlib import Path
import pandas as pd
import unicodedata

arabic_diacritics = re.compile(
    """
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """,
    re.VERBOSE,
)

arabic_punctuations = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
translator = str.maketrans("", "", punctuations_list)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, "", text)
    return text


def remove_accent(text):
    return "".join(
        (
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    )


def clean_dataset(df: pd.DataFrame, path: Path) -> pd.DataFrame:

    df["arabic"] = df["arabic"].str.lower()
    df["english"] = df["english"].str.lower()

    unicode_chars = ["\u200c", "\u200e", "\u200f", "\u202c"]
    for x in unicode_chars:
        df["arabic"] = df["arabic"].str.replace(x, " ").str.strip()
        df["english"] = df["english"].str.replace(x, " ").str.strip()

    df["arabic"] = df["arabic"].map(lambda x: x.translate(translator))
    df["english"] = df["english"].map(lambda x: x.translate(translator))

    df["arabic"] = df["arabic"].map(lambda x: " ".join(x.split()))
    df["english"] = df["english"].map(lambda x: " ".join(x.split()))

    df["arabic"] = df["arabic"].map(lambda x: normalize_arabic(x))
    df["arabic"] = df["arabic"].map(lambda x: remove_diacritics(x))

    df["english"] = df["english"].map(lambda x: remove_accent(x))

    df = df.loc[df["arabic"].map(len) > 1]
    df = df.loc[df["english"].map(len) > 1]

    df = df.dropna()

    df = df.drop_duplicates()

    df.sample(frac=1).reset_index(drop=True)

    df.to_csv(path, index=False, encoding="utf-8")

    return df
