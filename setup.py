from setuptools import setup, find_packages

setup(
    name="mas",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "python-telegram-bot",
        "requests",
        "python-dotenv"
    ],
)