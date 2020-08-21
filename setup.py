from setuptools import setup, find_packages

setup(
    name="names_transliteration",
    description="Names transliteration",
    packages=find_packages(exclude=[]),
    install_requires=["tensorflow==2.1.0", "pandas==1.1.0"],
    scripts=[],
    entry_points={
        "console_scripts": [
            "names-transliteration-get-data=transliteration.get_data:main",
            "names-transliteration-train=transliteration.train_nmt:main",
            "names-transliteration-transliterate=transliteration.transliterate_name:main",
        ]
    },
)
