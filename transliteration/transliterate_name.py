import sys
from argparse import ArgumentParser
from pathlib import Path

from transliteration import MODEL_DIR
from transliteration.model.nmt import get_model
from transliteration.model.save import load_keras_tokenizer_json, load_model_metadata
from transliteration.model.transliterate import transliterate


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("model_path", type=Path, nargs="?", default=MODEL_DIR)
    return parser.parse_args(args)


def run(name: str, model_path: Path) -> str:
    input_tokenizer = load_keras_tokenizer_json(model_path / "source_tokenizer.json")
    output_tokenizer = load_keras_tokenizer_json(model_path / "target_tokenizer.json")
    model_metadata = load_model_metadata(model_path / "model_metadata.json")
    encoder, _, decoder = get_model(
        model_metadata["vocab_inp_size"],
        model_metadata["vocab_tar_size"],
        model_metadata["embedding_dim"],
        model_metadata["units"],
        model_metadata["batch_size"],
    )
    encoder.load_weights((model_path / "encoder/checkpoint").as_posix())
    decoder.load_weights((model_path / "decoder/checkpoint").as_posix())

    output = transliterate(
        name=name,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        encoder=encoder,
        decoder=decoder,
        metadata=model_metadata,
    )

    return output


def main():
    args = parse_args(sys.argv[1:])
    name = args.name
    model_path = args.model_path
    _ = run(name, model_path)


if __name__ == "__main__":
    print(main())
