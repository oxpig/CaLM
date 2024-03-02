# CaLM
## The Codon adaptation Language Model

This repository encapsulates all code required to reproduce the results of the paper ["Codon language embeddings provide strong signals for use in protein engineering"](https://www.nature.com/articles/s42256-024-00791-0), by Carlos Outeiral and Charlotte M. Deane.


## Citation

If you use our work, please cite:

> Outeiral, Carlos, and Charlotte M. Deane. 
> *Codon language embeddings provide strong signals for use in protein engineering*
> __Nature Machine Intelligence__ 6.2 (2024): 170-179.


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
