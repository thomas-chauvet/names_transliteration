from setuptools import setup, find_packages

# Read dependencies from requirements file
with open("requirements.txt", "r") as requirements_file:
    dependencies = [line.rstrip("\n") for line in requirements_file]

setup(
    name="names_translation",
    description="Names translation",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=[]),
    install_requires=dependencies,
    scripts=[],
    # Optional
    entry_points={
        "console_scripts": [
            "names-translation-get-data=names_translation.get_data:main",
        ]
    },
)
