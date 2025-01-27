from setuptools import setup, find_packages

setup(
    name="mas",
    version="0.1.7",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "python-telegram-bot",
        "requests",
        "python-dotenv"
    ],
)