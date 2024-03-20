# setup.py
from setuptools import setup, find_packages

setup(
    name="geoToolbox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pymap3d"],
    author="thomas hu",
    author_email="thomas@graphopti.com",
    description="A simple module for geo coordinate conversion",
    keywords="geo coordinate conversion",
)
