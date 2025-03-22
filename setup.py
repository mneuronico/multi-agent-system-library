from setuptools import setup, find_packages

setup(
    name="mas",
    version="0.1.20",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "python-telegram-bot",
        "requests",
        "python-dotenv",
        'backports.zoneinfo; python_version < "3.9"',
        'tzdata; python_version < "3.9"'
    ],
)