#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="craft_ml",
    description="",
    version="0.2.4",
    package_dir={"": "src"},
    zip_safe=True,
    packages=find_packages('src'),
    install_requires=[
        "numpy>=1.19.1",
        "scikit_learn>=0.24.1",
        "matplotlib>=3.1.3",
        "pandas>=1.0.1",
        "streamlit>=0.75.0",
        # "dataclasses>=0.6",
        "xgboost>=0.90"
    ],
    python_requires='>=3.6',
    setup_requires=[],
    entry_points={
        "console_scripts": [
            "craft-ml=craft_ml:main"
        ]
    },
)
