from pathlib import Path
import pandas as pd
import unicodedata
from names_transliteration.cleaning import arabic


def remove_accent(text):
    return "".join(
        (
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    )


def prepare_data(df: pd.DataFrame, path: Path) -> pd.DataFrame:

    df["arabic"] = df["arabic"].map(
        lambda x: arabic.clean_name(x, start_token="", end_token="")[0]
    )
    df["english"] = (
        df["english"]
        .str.lower()
        .map(lambda x: " ".join(x.split()))
        .map(lambda x: remove_accent(x))
    )

    # Filter "names" with more than 1 character
    df = df.loc[df["arabic"].map(len) > 1]
    df = df.loc[df["english"].map(len) > 1]
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.sample(frac=1, random_state=777).reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df
