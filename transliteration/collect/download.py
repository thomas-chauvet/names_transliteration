from pathlib import Path
import pandas as pd


def data_ar2en_google_transliteration(output_dir: Path, filename: str = None):
    url = "https://raw.githubusercontent.com/google/transliteration/master/ar2en.txt"
    df = pd.read_table(url, sep="\t", names=["arabic", "english"])
    if filename is None:
        filename = "ar2en_google_transliteration.csv"
    df = df.loc[:, ["arabic", "english"]]
    df.to_csv(output_dir / filename, index=False, encoding="utf-8")
    return df


def data_ar2en_anetac(output_dir: Path, filename: str = None):
    url = "https://raw.githubusercontent.com/MohamedHadjAmeur/ANETAC/master/EN-AR%20NE/EN-AR%20Named-entities.txt"
    df = pd.read_table(url, sep=" ", names=["type", "english", "arabic"])
    df = df.loc[df["type"] == "PERSON", ["arabic", "english"]]
    if filename is None:
        filename = "ar2en_anetac.csv"
    df.to_csv(output_dir / filename, index=False, encoding="utf-8")
    return df


def data_ar2en_coling_2018(output_dir: Path, filename: str = None):
    url = "https://raw.githubusercontent.com/steveash/NETransliteration-COLING2018/master/data/wd_arabic.normalized.aligned.tokens"
    df = pd.read_table(url, sep="\t", names=["english", "arabic", "n"], quoting=3)
    df = df.dropna()
    df["arabic"] = df["arabic"].str.replace(" ", "")
    # filter row with a number in it
    df = df.loc[~df["arabic"].map(lambda x: any(map(str.isnumeric, x)))]
    # filter row with quote
    df = df.loc[~df["arabic"].str.contains('"')]
    df = df.loc[~df["english"].str.contains('"')]
    if filename is None:
        filename = "ar2en_coling_2018.csv"
    df = df.loc[:, ["arabic", "english"]]
    df.to_csv(output_dir / filename, index=False, encoding="utf-8")
    return df
