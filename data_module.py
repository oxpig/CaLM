"""Data modules for PyTorch Lightning."""

from argparse import Namespace
from typing import Optional

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from calm.alphabet import Alphabet
from calm.dataset import SequenceDataset
from calm.pipeline import (
    Pipeline,
    DataCollator,
    DataTrimmer,
    DataPadder,
    DataPreprocessor,
    CodonRandomizer,
    DataPreprocessorForDualData,
)


class CodonDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for manipulating FASTA files
    containing sequences of codons."""

    def __init__(self, args: Namespace, alphabet: Alphabet, data_dir: str,
            batch_size: int, random_seed: int = 42, test_size : float = 0.01):
        super().__init__()
        self.data_dir = data_dir
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.pipeline =  Pipeline([
            DataCollator(args, alphabet),
            DataTrimmer(args, alphabet),
            DataPadder(args, alphabet),
            DataPreprocessor(args, alphabet)
        ])

        self.train_data = None
        self.val_data = None

    def setup(self, stage: Optional[str] = None):
        dataset = SequenceDataset(self.data_dir, codon_sequence=True)
        self.train_data, self.val_data = train_test_split(dataset,
            test_size=self.test_size, shuffle=True, random_state=self.random_seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, num_workers=3,
            batch_size=self.batch_size, collate_fn=self.pipeline)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, num_workers=1,
            batch_size=self.batch_size, collate_fn=self.pipeline)
