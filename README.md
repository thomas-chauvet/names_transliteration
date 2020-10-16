# Names transliteration

In this repository you will find:
- a [dataset](https://raw.githubusercontent.com/thomas-chauvet/names_transliteration/master/data/clean/arabic_english.csv) 
(and associated code to build it) containing 
names in arabic characters and associated names in latin 
characters (english),
- a (google colab) notebook to train a 
[Neural Machine Translation](https://en.wikipedia.org/wiki/Neural_machine_translation) (NMT) model
based on [seq2seq](https://en.wikipedia.org/wiki/Seq2seq). The objective
of this model is to [transliterate](https://en.wikipedia.org/wiki/Transliteration) names
in arabic alphabet to latin alphabet. This task is also called 
[romanization](https://en.wikipedia.org/wiki/Romanization).

The model is trained thanks to Google Colab providing (free) GPU.

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

## Pre-trained models

A pre-trained model (arabic to latin characters) is stored on 
[dropbox](https://www.dropbox.com/s/leqc4k9c4hzfvi3/names-translation-model-2020-10-02.zip?dl=1).

## Library

Install library:
```bash
python setup.py install
```

## CLI

- `get-data`: Get data from 3 sources to get a training dataset.
- `get-pretrained-model`: Download pre-trained model for the task.
- `train-nmt-model`: Train an NMT model.
- `transliterate-name`: Transliterate a name in arabic in latin character.

## Python environment

Please refer to the `environment.yml` file for conda environment.

To create the environment with conda:
```bash
conda env create -f environment.yml
```

## ToDo

- add unit tests (and associated GH action)
- add streamlit app to use the model