# CaLM
## The Codon adaptation Language Model

This repository encapsulates all code required to reproduce the results of the paper ["Codon language embeddings provide strong signals for protein engineering"](https://www.biorxiv.org/content/10.1101/2022.12.15.519894v1.abstract), by Carlos Outeiral and Charlotte M. Deane.


## Installation

```python
git clone https://github.com/oxpig/CaLM
python setup.py install
```

## Usage

```python
from calm import CaLM

model = CaLM()
model.embed_sequence('ATGGTATAGAGGCATTGA')
```
