from pathlib import Path
from typing import Optional
import zipfile
import shutil
import tempfile
from uuid import uuid4

import typer
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from names_transliteration import APP_NAME, PRE_TRAINED_MODEL_URL
from names_transliteration.collect import download
from names_transliteration.collect.process import prepare_data
from names_transliteration.download import url_exists, download_from_url
from names_transliteration.model import (
    BATCH_SIZE,
    UNITS,
    EMBEDDING_DIM,
    NUM_EXAMPLES,
    EPOCHS,
    TEST_SIZE,
)
from names_transliteration.model.nmt import get_model
from names_transliteration.model.process import load_dataset
from names_transliteration.model.save import (
    load_keras_tokenizer_json,
    load_model_metadata,
    save_keras_tokenizer_json,
    save_model_metadata,
)
from names_transliteration.model.train import train
from names_transliteration.model.transliterate import transliterate

app = typer.Typer()


@app.command()
def get_data(raw_dir: Optional[Path] = None, clean_dir: Optional[Path] = None):
    """
    Get data from 3 sources to get a training dataset containing names in latin and arabic characters.

    You can specify:

    --raw-dir: path where to put "raw" downloaded files,

    --clean-dir: path where to put the concatenated and cleaned dataset.


    We use 3 datasets:

        - Google transliteration data. Example: *عادل; adel*,

        - ANETAC dataset. Example: *PERSON; Adel; اديل*. For this file we'll filter on *PERSON* only,

        - NETranliteration COLING 2018.

    These 3 datasets will give us a clean dataset containing names in arabic and
    corresponding names in latin alphabet (english).
    """
    app_dir = Path(typer.get_app_dir(APP_NAME))

    if not raw_dir:
        raw_dir = app_dir / "data" / "raw"

    if not clean_dir:
        clean_dir = app_dir / "data" / "clean"

    if not raw_dir.exists():
        typer.echo(f"Create '{raw_dir.absolute()}' directory.")
        raw_dir.mkdir(parents=True, exist_ok=True)

    if not clean_dir.exists():
        typer.echo(f"Create '{clean_dir.absolute()}' directory.")
        clean_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Download 'google transliteration' dataset...")
    df_google_transliteration = download.data_ar2en_google_transliteration(raw_dir)

    typer.echo("Download 'anetac' dataset...")
    df_anetac = download.data_ar2en_anetac(raw_dir)

    typer.echo("Download 'coling 2018' dataset...")
    df_coling = download.data_ar2en_coling_2018(raw_dir)

    df_arabic_english = pd.concat([df_google_transliteration, df_anetac, df_coling,])

    typer.echo("Prepare concatenated dataset...")
    _ = prepare_data(df_arabic_english, clean_dir / "arabic_english.csv")

    typer.echo(
        f"All done. Clean file is in {(clean_dir / 'arabic_english.csv').absolute()}"
    )


@app.command()
def train_nmt_model(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    units: int = UNITS,
    embedding_dim: int = EMBEDDING_DIM,
    num_examples: int = NUM_EXAMPLES,
    epochs: int = EPOCHS,
    test_size: float = TEST_SIZE,
):
    """
    Train an NMT model.

    You can specify:

    --dataset-path: "cleaned" dataset path.

    --model-path: path where to store model's output.

    --batch-size: batch size. Default: 64.

    --units: unit in the architecture. Default: 1024.

    --embedding-dim: embedding dimension in the neural network. Default: 256.

    --num-examples: limit the number of "examples"/lines to use. Default: take all.

    --epochs: Number of epochs. Default 10.

    --test-size: proportion of the dataset to use for testing. Default: 0.2.

    """
    app_dir = Path(typer.get_app_dir(APP_NAME))

    if not dataset_path:
        dataset_path = app_dir / "data" / "clean" / "arabic_english.csv"
        if not dataset_path.exists():
            raise Exception(
                f"Data has not been downloaded in {dataset_path.absolute()}"
            )

    if not model_path:
        model_path = app_dir / "model" / str(uuid4())

    typer.echo("Load dataset")
    source_tensor, target_tensor, source, target = load_dataset(
        dataset_path, num_examples
    )

    typer.echo("Creating training and validation sets")
    (
        source_tensor_train,
        source_tensor_val,
        target_tensor_train,
        target_tensor_val,
    ) = train_test_split(source_tensor, target_tensor, test_size=test_size)

    BUFFER_SIZE = len(source_tensor_train)
    typer.echo(f"BUFFER_SIZE: {BUFFER_SIZE}")
    STEPS_PER_EPOCH = len(source_tensor_train) // batch_size
    typer.echo(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}")
    VOCAB_INP_SIZE = len(source.word_index) + 1
    typer.echo(f"VOCAB_INP_SIZE: {VOCAB_INP_SIZE}")
    VOCAB_TAR_SIZE = len(target.word_index) + 1
    typer.echo(f"VOCAB_TAR_SIZE: {VOCAB_TAR_SIZE}")

    typer.echo("Tensorflow dataset batch")
    dataset = tf.data.Dataset.from_tensor_slices(
        (source_tensor_train, target_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    typer.echo("Instanciate encoder, attention and decoder")
    encoder, attention_layer, decoder = get_model(
        VOCAB_INP_SIZE, VOCAB_TAR_SIZE, embedding_dim, units, batch_size
    )

    typer.echo("Train model")
    encoder, decoder = train(
        dataset,
        encoder,
        decoder,
        target,
        STEPS_PER_EPOCH,
        epochs,
        checkpoint_dir=model_path / "training_checkpoints",
    )

    metadata = {
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "units": units,
        "vocab_inp_size": len(source.word_index) + 1,
        "vocab_tar_size": len(target.word_index) + 1,
        "max_length_source": source_tensor.shape[1],
        "max_length_target": target_tensor.shape[1],
    }

    typer.echo("Save models weights")
    encoder.save_weights(
        (model_path / "encoder/checkpoint").as_posix(), save_format="tf"
    )
    decoder.save_weights(
        (model_path / "decoder/checkpoint").as_posix(), save_format="tf"
    )

    typer.echo("Save tokenizers")
    save_keras_tokenizer_json(source, model_path / "source_tokenizer.json")
    save_keras_tokenizer_json(target, model_path / "target_tokenizer.json")

    typer.echo("Save model's metadata")
    save_model_metadata(metadata, model_path / "model_metadata.json")

    typer.echo(f"Everyting saved under {model_path.absolute()}")


@app.command()
def get_pretrained_model(model_path: Optional[Path] = None):
    """
    Download pre-trained model for the task.

    You can specify:

    --model-path: path where to store downloaded model.
    """

    if not model_path:
        model_path = Path(typer.get_app_dir(APP_NAME)) / "model"

    if model_path.exists():
        typer.echo(f"{model_path.absolute()} already exists.")

    if not url_exists(PRE_TRAINED_MODEL_URL):
        raise Exception(f"{PRE_TRAINED_MODEL_URL} does nost exist or is unreachable.")

    temp_directory = Path(tempfile.gettempdir())
    temp_file = temp_directory / "model.zip"

    try:
        download_from_url(url=PRE_TRAINED_MODEL_URL, dst=temp_file)
    except FileExistsError:
        typer.echo(f"Pre-trained model already present in {temp_file.absolute()}")
        raise typer.Exit()

    typer.echo("Unzip model")
    with zipfile.ZipFile(temp_file, "r") as zip_ref:
        zip_ref.extractall(temp_file.parent)

    shutil.move(
        (temp_file.parent / "names-translation-model-2020-10-02").as_posix(),
        model_path.as_posix(),
    )
    typer.echo(f"Model is in {model_path.absolute()}")


@app.command()
def transliterate_name(
    name: str, model_path: Optional[Path] = None, verbose: bool = False
):
    """
    Transliterate a name in arabic in latin character.

    You must specify:

    name: name in arabic.

    ```
    names-transliteration transliterate-name محمد
    ```

    You can specify:

    --model-path: path where the model is stored.
    --verbose: sould output verbose information? Default: false.
    """

    if verbose:
        typer.echo("Load Keras tokenizers")

    if not model_path:
        model_path = Path(typer.get_app_dir(APP_NAME)) / "model"

    if not model_path.exists():
        typer.echo(f"{model_path.absolute()} does not exist.")

    output_tokenizer = load_keras_tokenizer_json(model_path / "target_tokenizer.json")
    input_tokenizer = load_keras_tokenizer_json(model_path / "source_tokenizer.json")
    if verbose:
        typer.echo("Load model's metadata")
    model_metadata = load_model_metadata(model_path / "model_metadata.json")

    if verbose:
        typer.echo("Prepare model")
    encoder, _, decoder = get_model(
        model_metadata["vocab_inp_size"],
        model_metadata["vocab_tar_size"],
        model_metadata["embedding_dim"],
        model_metadata["units"],
        model_metadata["batch_size"],
    )

    if verbose:
        typer.echo("Load encoder/decoder weights")
    encoder.load_weights((model_path / "encoder/checkpoint").as_posix())
    decoder.load_weights((model_path / "decoder/checkpoint").as_posix())

    if verbose:
        typer.echo("Transliterate")
    output = transliterate(
        name=name,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        encoder=encoder,
        decoder=decoder,
        metadata=model_metadata,
    )

    typer.echo(output)


if __name__ == "__main__":
    app()
