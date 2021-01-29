#!/usr/bin/env python3

from setuptools import setup


setup(
    name="craft_ml",
    description="",
    version="0.1",
    package_dir={"": "src"},
    zip_safe=True,
    packages=["craft_ml"],
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pathos"
    ],
    setup_requires=[],
    entry_points={
        "console_scripts": [
            "craft-ml=craft_ml:main"
        ]
    },
)
