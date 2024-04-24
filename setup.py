import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

description = 'Read CG-6 data file and simple proc of ties'

# get version
with open('version.txt', mode='r', encoding='utf8') as fh:
    version = fh.read()

scripts  = [os.path.join('scripts', f) for f in os.listdir('scripts') if f.endswith('.py')]
with open('requirements.txt', mode='r', encoding='utf-8') as fh:
    install_requires = [line.split().pop(0) for line in fh.read().splitlines()]

setup(
    name = 'rgrav_proc',
    version = version,
    author = 'Roman Sermiagin',
    author_email = 'roman.sermiagin@gmail.com',
    description = description,
    license = "MIT License",
    keywords = 'scintrex cg-6 gravimeter',
    # url = "",
    packages=find_packages(),
    long_description=read('README.md'),
    # classifiers=[
    #     "Development Status :: 1 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    install_requires = install_requires,
    scripts=scripts,
)
