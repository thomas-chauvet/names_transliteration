from pathlib import Path
from argparse import ArgumentParser
import sys
import pandas as pd
from names_translation.collect import download
from names_translation.collect.cleaning import clean_dataset


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument(
        "raw_dir", type=Path, nargs="?", default=Path.cwd().parent / "data" / "raw"
    )
    parser.add_argument(
        "clean_dir", type=Path, nargs="?", default=Path.cwd().parent / "data" / "clean"
    )
    return parser.parse_args(args)


def run(raw_dir: Path, clean_dir: Path):

    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    df_arabic_english = pd.concat(
        [
            download.data_ar2en_google_transliteration(raw_dir),
            download.data_ar2en_anetac(raw_dir),
            download.data_ar2en_coling_2018(raw_dir),
        ]
    )

    df_arabic_english = clean_dataset(
        df_arabic_english, clean_dir / "arabic_english.csv"
    )

    return df_arabic_english


def main():
    args = parse_args(sys.argv[1:])
    raw_dir = args.raw_dir
    clean_dir = args.clean_dir
    _ = run(raw_dir, clean_dir)


if __name__ == "__main__":
    main()
