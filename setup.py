from setuptools import setup, find_packages

setup(
    name="endure",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
    ],
    python_requires=">=3.8",
) 