from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='CaLM',
    version='0.1.1',
    description='CaLM: the Codon adaptation Language Model',
    license='BSD 3-clause license',
    maintainer='Carlos Outeiral',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email='carlos.outeiral@stats.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('calm', 'calm.*')),
    install_requires=[
        'numpy',
        'requests',
        'torch>=1.6',
        'biopython',
        'einops',
        'rotary_embedding_torch'
    ],
)
