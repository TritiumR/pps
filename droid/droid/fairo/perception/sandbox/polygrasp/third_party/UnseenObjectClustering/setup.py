from setuptools import setup, find_packages

setup(
    name="uoc",
    version="0.1",
    packages=find_packages(where="lib/"),
    package_dir={"": "lib/"},
    install_requires=["easydict"]
)
