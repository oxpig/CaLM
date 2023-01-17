"""Classes to deal with codon sequences."""

import abc
from Bio.Seq import Seq
from typing import Union, List


def _split_into_codons(seq: str):
    """Yield successive 3-letter chunks of a string/sequence."""
    for i in range(0, len(seq), 3):
        yield seq[i:i + 3]

class Sequence(abc.ABC):
    """Abstract base class for sequence data."""

    @property
    def seq(self):
        return self._seq 

    @property
    def tokens(self):
        return self._seq.split()

    def _sanitize(self, tokens: List[str]):
        return [x.strip() for x in tokens
            if x.strip() != '']


class CodonSequence(Sequence):
    """Class containing a sequence of codons.

    >>> seq = CodonSequence('ATGGCGCTAAAGCGGATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']

    >>> seq = CodonSequence('ATG GCG CTA AAG CGG ATC')
    >>> seq.tokens
    ['<cls>', 'AUG', 'GCG', 'CUA', 'AAG', 'CGG', 'AUC', '<eos>']
    """

    def __init__(self, seq_: Union[str, Seq]):
        super().__init__()
        seq = str(seq_)
        _tokens = ['<cls>'] \
            + list(_split_into_codons(seq.replace('T', 'U').replace(' ', ''))) \
            + ['<eos>']
        _tokens = self._sanitize(_tokens)
        self._seq = ' '.join(_tokens)

