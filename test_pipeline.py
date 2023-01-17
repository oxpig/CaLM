
from collections import namedtuple

from calm.alphabet import Alphabet
from calm.sequence import CodonSequence
from calm.pipeline import (
    Pipeline,
    PipelineInput,
    DataCollator,
    DataTrimmer,
    DataPadder,
    DataPreprocessor,
)


def fake_args():
    Args = namedtuple('args', [
        'mask_proportion',
        'max_positions',
        'mask_percent',
        'leave_percent'
    ])
    return Args(mask_proportion=.25, max_positions=10,
        mask_percent=.8, leave_percent=.1)

def test_DataCollator_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_collator = DataCollator(args, alphabet)

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    input_ = PipelineInput(sequence=[seq1, seq2])
    output = data_collator(input_)

    assert output.ground_truth[0] == seq1.seq
    assert output.sequence[0].split().count('<mask>') == int(len(seq1.tokens) * .25 * .8)
    assert output.target_mask[0].sum() == int(len(seq1.tokens) * .25)

def test_DataTrimmer_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_trimmer = Pipeline([
        DataCollator(args, alphabet),
        DataTrimmer(args, alphabet)
    ])

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    output = data_trimmer([seq1, seq2])

def test_DataPadder_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_padder = Pipeline([
        DataCollator(args, alphabet),
        DataTrimmer(args, alphabet),
        DataPadder(args, alphabet),
    ])

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    output = data_padder([seq1, seq2])

def test_DataPreprocessor_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_preprocessor = Pipeline([
        DataCollator(args, alphabet),
        DataTrimmer(args, alphabet),
        DataPadder(args, alphabet),
        DataPreprocessor(args, alphabet)
    ])

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    output = data_preprocessor([seq1, seq2])

