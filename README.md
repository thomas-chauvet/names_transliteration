# Names transliteration

Arabic/English names dataset.

## Dataset

We use 3 datasets:
*   Google transliteration dataset from repository on [github](https://github.com/google/transliteration). Example: *عادل	adel*
*   ANETAC dataset on [github](https://github.com/MohamedHadjAmeur/ANETAC). Example: *PERSON Adel اديل*. For this dataset we'll filter on *PERSON* only,
*   NETranliteration COLING 2018 dataset on [github](https://github.com/steveash/NETransliteration-COLING2018/blob/master/data/wd_arabic.normalized.aligned.tokens).

These 3 datasets will give us a clean dataset containing arabic and corresponding english names.

To get the final dataset you can run the script `get_data.py`.

## Library

Install library:
```bash
python setup.py install
```

Get data from github in raw folder:
```bash
names-translation-get-data data/raw data/clean
```

## Python environment

Please refer to the `environment.yml` file for conda environment or to the
`requirements.txt` for virtual environment.
