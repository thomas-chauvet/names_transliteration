import sys
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from transliteration import DATA_DIR, MODEL_DIR
from transliteration.model import (
    BATCH_SIZE,
    UNITS,
    EMBEDDING_DIM,
    TEST_SIZE,
    EPOCHS,
)
from transliteration.model.nmt import get_model
from transliteration.model.process import load_dataset
from transliteration.model.save import save_keras_tokenizer_json, save_model_metadata
from transliteration.model.train import train

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DATA_DIR / "clean" / "arabic_english.csv",
    )
    parser.add_argument("model_path", type=Path, nargs="?", default=MODEL_DIR)
    parser.add_argument("batch_size", type=int, nargs="?", default=BATCH_SIZE)
    parser.add_argument("units", type=int, nargs="?", default=UNITS)
    parser.add_argument("embedding_dim", type=int, nargs="?", default=EMBEDDING_DIM)
    parser.add_argument("num_examples", type=int, nargs="?", default=None)
    parser.add_argument("epochs", type=int, nargs="?", default=EPOCHS)
    parser.add_argument("test_size", type=float, nargs="?", default=TEST_SIZE)

    return parser.parse_args(args)


def run(
    dataset_path: Path,
    model_path: Path,
    batch_size: int,
    units: int,
    embedding_dim: int,
    num_examples: int,
    epochs: int,
    test_size: float,
):
    logger.info("Load dataset")
    source_tensor, target_tensor, source, target = load_dataset(
        dataset_path, num_examples
    )

    logger.info("Creating training and validation sets")
    (
        source_tensor_train,
        source_tensor_val,
        target_tensor_train,
        target_tensor_val,
    ) = train_test_split(source_tensor, target_tensor, test_size=test_size)

    BUFFER_SIZE = len(source_tensor_train)
    logger.info(f"BUFFER_SIZE: {BUFFER_SIZE}")
    STEPS_PER_EPOCH = len(source_tensor_train) // batch_size
    logger.info(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}")
    VOCAB_INP_SIZE = len(source.word_index) + 1
    logger.info(f"VOCAB_INP_SIZE: {VOCAB_INP_SIZE}")
    VOCAB_TAR_SIZE = len(target.word_index) + 1
    logger.info(f"VOCAB_TAR_SIZE: {VOCAB_TAR_SIZE}")

    logger.info("Tensorflow dataset batch")
    dataset = tf.data.Dataset.from_tensor_slices(
        (source_tensor_train, target_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    logger.info("Instanciate encoder, attention and decoder")
    encoder, attention_layer, decoder = get_model(
        VOCAB_INP_SIZE, VOCAB_TAR_SIZE, embedding_dim, units, batch_size
    )

    logger.info("Train model")
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

    logger.info("Save models weights")
    encoder.save_weights(
        (model_path / "encoder/checkpoint").as_posix(), save_format="tf"
    )
    decoder.save_weights(
        (model_path / "decoder/checkpoint").as_posix(), save_format="tf"
    )

    logger.info("Save tokenizers")
    save_keras_tokenizer_json(source, model_path / "source_tokenizer.json")
    save_keras_tokenizer_json(target, model_path / "target_tokenizer.json")

    logger.info("Save model's metadata")
    save_model_metadata(metadata, model_path / "model_metadata.json")


def main():
    args = parse_args(sys.argv[1:])
    dataset_path = args.dataset_path
    model_path = args.model_path
    batch_size = args.batch_size
    units = args.units
    embedding_dim = args.embedding_dim
    num_examples = args.num_examples
    epochs = args.epochs
    test_size = args.test_size
    _ = run(
        dataset_path,
        model_path,
        batch_size,
        units,
        embedding_dim,
        num_examples,
        epochs,
        test_size,
    )


if __name__ == "__main__":
    main()
