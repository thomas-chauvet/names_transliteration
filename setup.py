from setuptools import setup, find_packages

setup(
    name="names_transliteration",
    version="0.0.1",
    description="Names transliteration from arabic to latin characters.",
    packages=find_packages(exclude=[]),
    python_requires=">=3.5, <4",
    install_requires=[
        "tensorflow==2.1.0",
        "pandas==1.1.0",
        "scikit-learn==0.23.2",
        "tqdm==4.50.0",
        "typer==0.3.1",
    ],
    scripts=[],
    entry_points={
        "console_scripts": ["names-transliteration=names_transliteration.main:app"]
    },
)
