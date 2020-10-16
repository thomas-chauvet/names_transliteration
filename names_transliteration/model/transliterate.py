from typing import Dict, Any

import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from names_transliteration.cleaning.arabic import clean_name
from names_transliteration.model.nmt import Encoder, Decoder


def check_letter_in_tokenizer(name: str, input_tokenizer: Tokenizer):
    letter_not_in_tokenizer = [
        l for l in name if l not in input_tokenizer.index_word.values()
    ]
    if len(letter_not_in_tokenizer) > 0:
        raise Exception(
            f"Some letters are not in tokenizer's characters ({', '.join(letter_not_in_tokenizer)})"
        )
    return True


def evaluate(name, input_tokenizer, output_tokenizer, encoder, decoder, metadata):

    inputs = [input_tokenizer.word_index[i] for i in name]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=metadata["max_length_source"], padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, metadata["units"]))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([output_tokenizer.word_index["!"]], 0)

    for t in range(metadata["max_length_target"]):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += output_tokenizer.index_word[predicted_id] + " "

        if output_tokenizer.index_word[predicted_id] == "$":
            return result, name

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    result = result.split("?")[0].replace(" ", "")

    return result


def transliterate(
    name: str,
    input_tokenizer: Tokenizer,
    output_tokenizer: Tokenizer,
    encoder: Encoder,
    decoder: Decoder,
    metadata: Dict[str, Any],
) -> str:
    names = clean_name(name)
    _ = check_letter_in_tokenizer(name, input_tokenizer)
    result = " ".join(
        [
            evaluate(
                name, input_tokenizer, output_tokenizer, encoder, decoder, metadata
            )
            for name in names
        ]
    )
    return result
