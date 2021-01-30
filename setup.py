#!/usr/bin/env python3

from setuptools import setup


setup(
    name="craft_ml",
    description="",
    version="0.2",
    package_dir={"": "src"},
    zip_safe=True,
    packages=["craft_ml"],
    install_requires=[
        "numpy==1.18.1",
        "scikit_learn==0.24.1",
        "matplotlib==3.1.3",
        "streamlit",
        "pandas==1.0.1",
        "streamlit==0.75.0",
    ],
    python_requires='>=3.7',
    setup_requires=[],
    entry_points={
        "console_scripts": [
            "craft-ml=craft_ml:main"
        ]
    },
)
