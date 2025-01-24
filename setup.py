from setuptools import setup, find_packages

setup(
    name="mas",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai>=1.58.1",
        "groq>=0.13.1",
        "google-generativeai>=0.8.3",
        "anthropic>=0.45.0",
        "python-telegram-bot>=21.7"
    ],
)