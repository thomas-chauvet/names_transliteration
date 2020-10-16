from names_transliteration.model.nmt.attention import BahdanauAttention
from names_transliteration.model.nmt.decoder import Decoder
from names_transliteration.model.nmt.encoder import Encoder


def get_model(
    vocab_inp_size: int,
    vocab_tar_size: int,
    embedding_dim: int,
    units: int,
    batch_sz: int,
):
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_sz)
    attention = BahdanauAttention(10)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_sz)
    return encoder, attention, decoder
