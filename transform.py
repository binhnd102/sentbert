import numpy as np
from tokenizer import tokenizer
import random

non_special_vocabs = tokenizer.get_non_special_vocab()
vocabs = tokenizer.get_vocab()
pair_matrix = np.load("data/chotot/pair_matrix.npy")

def remove_chars(sent, p=0.05):
    return "".join([sent[i] for i, val in enumerate(np.random.binomial(1, p, len(sent))==0) if val])


def replace_chars(sent, p=0.05):
    random_words = np.random.choice(non_special_vocabs, len(sent))
    return "".join([sent[i] if val else random_words[i] for i, val in enumerate(np.random.binomial(1, p, len(sent))==0)])


def gen_pair(c):
    random_c = random.choice(non_special_vocabs)
    if c not in tokenizer.stoi:
        return c + random_c
    p = pair_matrix[tokenizer.stoi[c]] / pair_matrix[tokenizer.stoi[c]].sum()
    next_c = np.random.choice(vocabs,1,p=p)[0]
    if (next_c not in tokenizer.stoi) or (next_c not in non_special_vocabs):
        return c + random_c
    return c + next_c


def insert_chars(sent, p=0.05):
    return "".join([sent[i] if val else gen_pair(sent[i]) for i, val in enumerate(np.random.binomial(1, p, len(sent))==0)])


def transform_sent(sent):
    p = np.random.rand()
    if p < 0.3:
        return remove_chars(sent)
    elif p < 0.6:
        return replace_chars(sent)
    else:
        return insert_chars(sent)