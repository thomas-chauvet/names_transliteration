import time
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from transliteration.model import EPOCHS, BATCH_SIZE
from transliteration.model.nmt import Encoder, Decoder
from transliteration.model.nmt.loss import loss_function

import logging

logger = logging.getLogger(__name__)


def train(
    dataset,
    encoder: Encoder,
    decoder: Decoder,
    target_tokenizer,
    steps_per_epoch: int,
    epochs: int = EPOCHS,
    checkpoint_dir: Path = Path("./training_checkpoints"),
) -> Tuple[Encoder, Decoder]:
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_prefix = checkpoint_dir / "ckpt"
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, encoder=encoder, decoder=decoder
    )

    @tf.function
    def train_step(source, target, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(source, enc_hidden)

            dec_hidden = enc_hidden

            dec_source = tf.expand_dims(
                [target_tokenizer.word_index["!"]] * BATCH_SIZE, 1
            )

            # Teacher forcing - feeding the target as the next source
            for t in range(1, target.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_source, dec_hidden, enc_output)

                loss += loss_function(target[:, t], predictions)

                # using teacher forcing
                dec_source = tf.expand_dims(target[:, t], 1)

        batch_loss = loss / int(target.shape[1])

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    for epoch in range(epochs):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (input, output)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(input, output, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(
                    "Epoch {} Batch {} Loss {:.4f}".format(
                        epoch + 1, batch, batch_loss.numpy()
                    )
                )
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        logger.info(
            "Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch)
        )
        logger.info("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    return encoder, decoder
