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
from names_transliteration.model.transliterate import (
    transliterate,
    LettersNotInTokenizerException,
)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())

MODEL_PATH = Path(typer.get_app_dir(APP_NAME)) / "model"


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.

    @st.cache(show_spinner=False)
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

    @st.cache(show_spinner=False, allow_output_mutation=True)
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

    with st.spinner("Downloading pre-trained models..."):
        download_pre_trained_model()

    with st.spinner("Load model's weights..."):
        (
            output_tokenizer,
            input_tokenizer,
            model_metadata,
            encoder,
            decoder,
        ) = load_model()

    name_input = st.text_input(label="Enter a names in arabic characters").replace(
        "\u200e", ""
    )
    logger.info(f"name_input {name_input}")
    if len(name_input) > 0:
        try:
            with st.spinner("Transliterating..."):
                transliterated = transliterate(
                    name=name_input,
                    input_tokenizer=input_tokenizer,
                    output_tokenizer=output_tokenizer,
                    encoder=encoder,
                    decoder=decoder,
                    metadata=model_metadata,
                )
            st.success("Done!")
            logger.info(f"transliterated {transliterated}")
            st.text("Transliterated name is:")
            st.title(transliterated)
        except LettersNotInTokenizerException:
            st.error(
                "Some letters are not handled properly. Probably not arabic character."
            )
        except Exception as e:
            logger.error(e)


def main():
    st.title("Romanization: Transliteration from arabic to latin characters")
    st.sidebar.title("About")
    st.sidebar.markdown(
        """
    **Notes**: Only arabic characters can be used with the app. Other characters
    will raise an error.
    """
    )
    with (Path(__file__).parent / "README.md").open(encoding="utf-8") as f:
        about_md = f.read()
    st.sidebar.markdown(about_md.split("## Web application - Streamlit")[0])
    st.sidebar.markdown(
        """
    
    
    [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
    
    [![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://colab.research.google.com/github/thomas-chauvet/names_transliteration/blob/master/arabic_to_english_names_transliteration_with_nmt_and_attention.ipynb)
    
    [![GitHub](https://img.shields.io/badge/github-%23100000.svg?&style=for-the-badge&logo=github&logoColor=white)](https://github.com/thomas-chauvet/)
    
    [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/ChauvetThomas.svg?style=social&label=Follow%20%40ChauvetThomas)](https://twitter.com/ChauvetThomas)
    [![Linkedin](https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/thomaschauvet/)
    """
    )
    run_the_app()


if __name__ == "__main__":
    main()
