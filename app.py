import logging
import shutil
import tempfile
import zipfile

import streamlit as st
from pathlib import Path

import typer

from names_transliteration import APP_NAME, PRE_TRAINED_MODEL_URL
from names_transliteration.download import url_exists, download_from_url
from names_transliteration.model.nmt import get_model
from names_transliteration.model.save import (
    load_keras_tokenizer_json,
    load_model_metadata,
)
from names_transliteration.model.transliterate import transliterate, LettersNotInTokenizerException

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())

MODEL_PATH = Path(typer.get_app_dir(APP_NAME)) / "model"


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.

    @st.cache(show_spinner=True)
    def download_pre_trained_model():
        if not url_exists(PRE_TRAINED_MODEL_URL):
            raise Exception(
                f"{PRE_TRAINED_MODEL_URL} does nost exist or is unreachable."
            )

        logger.info("Create temp dir to download model")
        temp_directory = Path(tempfile.gettempdir())
        temp_file = temp_directory / "model.zip"

        pre_trained_model_downloaded = False
        try:
            logger.info("Download pre-trained model...")
            download_from_url(url=PRE_TRAINED_MODEL_URL, dst=temp_file)
            logger.info("Pre-trained model downloaded.")
        except FileExistsError:
            pre_trained_model_downloaded = True

        if not pre_trained_model_downloaded:
            logger.info("Extract model")
            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                zip_ref.extractall(temp_file.parent)

            logger.info("Move unzip files.")
            shutil.move(
                (temp_file.parent / "names-translation-model-2020-10-02").as_posix(),
                MODEL_PATH.as_posix(),
            )

    @st.cache(show_spinner=True, allow_output_mutation=True)
    def load_model():
        logger.info("Load output_tokenizer...")
        output_tokenizer = load_keras_tokenizer_json(
            MODEL_PATH / "target_tokenizer.json"
        )
        logger.info("output_tokenizer loaded.")
        logger.info("Load input_tokenizer...")
        input_tokenizer = load_keras_tokenizer_json(
            MODEL_PATH / "source_tokenizer.json"
        )
        logger.info("input_tokenizer loaded.")
        logger.info("Load model_metadata...")
        model_metadata = load_model_metadata(MODEL_PATH / "model_metadata.json")
        logger.info("model_metadata loaded.")
        logger.info("Load encoder/decoder.")
        encoder, _, decoder = get_model(
            model_metadata["vocab_inp_size"],
            model_metadata["vocab_tar_size"],
            model_metadata["embedding_dim"],
            model_metadata["units"],
            model_metadata["batch_size"],
        )
        encoder.load_weights((MODEL_PATH / "encoder/checkpoint").as_posix())
        decoder.load_weights((MODEL_PATH / "decoder/checkpoint").as_posix())
        logger.info("Encoder/decoder loaded")
        return output_tokenizer, input_tokenizer, model_metadata, encoder, decoder

    download_pre_trained_model()
    output_tokenizer, input_tokenizer, model_metadata, encoder, decoder = load_model()

    name_input = st.text_input(label="Enter a names in arabic characters")
    logger.info(f"name_input {name_input}")
    name = name_input.strip('\u200e')
    logger.info(f"name {name}")
    if len(name) > 0:
        try:
            transliterated = transliterate(
                name=name,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
                encoder=encoder,
                decoder=decoder,
                metadata=model_metadata,
            )
            logger.info(f"transliterated {transliterated}")
            st.text("Transliterated name is:")
            st.title(transliterated)
        except LettersNotInTokenizerException:
            st.error("Some letters are not handled properly. Probably not arabic character.")
        except Exception as e:
            logger.error(e)




def main():
    st.title("Romanization: Transliteration from arabic to latin characters")
    run_the_app()


if __name__ == "__main__":
    main()
