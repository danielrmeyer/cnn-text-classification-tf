from setuptools import setup, find_packages

setup(
    name="cnn-text-classification-tf",
    version="0.0.1",
    description="CNN Text Classification with tensorflow",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=["numpy", "tensorflow"],
)
