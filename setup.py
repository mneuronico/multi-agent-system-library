from setuptools import setup, find_packages

setup(
    name="mas",
    version="0.1.11",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "python-telegram-bot",
        "requests",
        "python-dotenv"
    ],
)