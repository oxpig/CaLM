"""Module to use CaLM as a pretrained model."""

import os
import pickle
import requests
from typing import Optional, Union, List

import torch
from .alphabet import Alphabet
from .sequence import CodonSequence
from .model import ProteinBertModel


class ArgDict:
    def __init__(self, d):
        self.__dict__ = d

_ARGS = {
    'max_positions': 1024,
    'batch_size': 46,
    'accumulate_gradients': 40,
    'mask_proportion': 0.25,
    'leave_percent': 0.10,
    'mask_percent': 0.80,
    'warmup_steps': 1000,
    'weight_decay': 0.1,
    'lr_scheduler': 'warmup_cosine',
    'learning_rate': 4e-4,
    'num_steps': 121000,
    'num_layers': 12,
    'embed_dim': 768,
    'attention_dropout': 0.,
    'logit_bias': False,
    'rope_embedding': True,
    'ffn_embed_dim': 768*4,
    'attention_heads': 12
}
ARGS = ArgDict(_ARGS)


class CaLM:
    """Module to use the Codon adaptation Language Model (CaLM)
    as published in C. Outeiral and C. M. Deane, "Codon language
    embeddings provide strong signals for protein engineering",
    bioRxiv (2022), doi: 10.1101/2022.12.15.519894."""

    def __init__(self, args: dict=ARGS, weights_file: Optional[str] = None) -> None:
        if weights_file is None:
            model_folder = os.path.join(os.path.dirname(__file__), 'calm_weights')
            weights_file = os.path.join(model_folder, 'calm_weights.ckpt')
            if not os.path.exists(weights_file):
                print('Downloading model weights...')
                os.makedirs(model_folder, exist_ok=True)
                url = 'http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl'
                with open(weights_file, 'wb') as handle:
                    handle.write(requests.get(url).content)

        self.alphabet = Alphabet.from_architecture('CodonModel')
        self.model = ProteinBertModel(args, self.alphabet)
        self.bc = self.alphabet.get_batch_converter()

        with open(weights_file, 'rb') as handle:
            state_dict = pickle.load(handle)
            self.model.load_state_dict(state_dict)

    def __call__(self, x):
        return self.model(x)

    def embed_sequence(self, sequence: Union[str, CodonSequence], average: bool = True) -> torch.Tensor:
        """Embeds an individual sequence using CaLM. If the ``average''
        flag is True, then the representation is averaged over all
        possible odons, providing a vector representation of the
        sequence."""
        if isinstance(sequence, str):
            seq = CodonSequence(sequence)
        elif isinstance(sequence, CodonSequence):
            seq = sequence
        else:
            raise ValueError('Input sequence must be string or CodonSequence.')

        tokens = self.tokenize(seq)
        repr_ = self.model(tokens, repr_layers=[12])['representations'][12]
        if average:
            return repr_.mean(axis=1)
        else:
            return repr_

    def embed_sequences(self, sequences: List[Union[str, CodonSequence]]) -> torch.Tensor:
        """Embeds a set of sequences using CaLM."""
        return torch.cat([self.embed_sequence(seq, average=True) for seq in sequences], dim=0)

    def tokenize(self, seq: CodonSequence) -> torch.Tensor:
        assert isinstance(seq, CodonSequence), 'seq must be CodonSequence'
        _, _, tokens = self.bc([('', seq.seq)])
        return tokens

