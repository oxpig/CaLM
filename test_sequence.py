
from calm.sequence import CodonSequence


def test_CodonSequence():
    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA')
    seq2 = CodonSequence('AUGGGACGCUUUUACCAAAUGGGACGCUUUUACCAAUAA')
    assert seq1.tokens == seq2.tokens
    assert seq1.seq == seq2.seq

