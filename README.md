# Names transliteration

In this repository you will find:
- a dataset (and associated code to build it) containing 
names in arabic characters and associated names in latin 
characters (english),
- a (google colab) notebook to train a 
[Neural Machine Translation](https://en.wikipedia.org/wiki/Neural_machine_translation) (NMT) model
based on [seq2seq](https://en.wikipedia.org/wiki/Seq2seq). The objective
of this model is to [transliterate](https://en.wikipedia.org/wiki/Transliteration) names
in arabic alphabet to latin alphabet. This task is also called 
[romanization](https://en.wikipedia.org/wiki/Romanization).

The model is trained thanks to google colab provided (free) GPU.

The model is based on Tensorflow tutorial 
[NMT with attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention).

## Data

We use 3 datasets:
*   [Google transliteration data](https://github.com/google/transliteration/blob/master/ar2en.txt).
Example: *عادل; adel*
*   [ANETAC dataset](https://github.com/MohamedHadjAmeur/ANETAC/blob/master/EN-AR%20NE/EN-AR%20Named-entities.txt). 
Example: *PERSON; Adel; اديل*. For this file we'll filter on *PERSON* only,
*   [NETranliteration COLING 2018](https://github.com/steveash/NETransliteration-COLING2018/blob/master/data/wd_arabic.normalized.aligned.tokens).

These 3 datasets will give us a clean dataset containing names in arabic and 
corresponding names in latin alphabet (english).

To get the final dataset you can run the script `get_data.py`.

## Library

Install library:
```bash
python setup.py install
```

Get data from github in raw folder:
```bash
names-transliteration-get-data data/raw data/clean
```

Train NMT model:
```bash
names-transliteration-train data/clean/arabic_english.csv model/ \
    64 1024 256 500 2 0.2
```

Evaluate a name (for instance "Mohammed"):
```bash
names-transliteration-transliterate محمد‎ model/
```

## Python environment

Please refer to the `environment.yml` file for conda environment.