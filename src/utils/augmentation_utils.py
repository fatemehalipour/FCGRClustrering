import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import CGR_utils
from Bio.Seq import Seq
from tqdm.auto import tqdm


def plot_random_fcgr(df,
                     k=6,
                     random_seed=42):
    random.seed(random_seed)
    fcgr = CGR_utils.FCGR(k=k)

    # Plot the image wiht matplotlib
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3
    for i in range(1, rows * cols + 1):
        random_idx = random.randint(0, len(df))
        seq, label = df.sequence[random_idx], df.label[random_idx]

        # Generate chaos for random sequence and normalize it
        chaos = fcgr(seq)
        chaos = chaos / len(seq)

        fig.add_subplot(rows, cols, i)
        plt.imshow(1 - chaos, cmap="gray")
        plt.title(label)
        plt.axis(False);


def mutation(seq: str,
             transition: bool,
             transversion: bool,
             transition_prob: float = 0.0,
             transversion_prob: float = 0.0) -> str:
    """
    Mutate the input sequence using transitions and transversions.
    Transition mutations occur when a pyrimidine base (i.e., thymine [T] or
    cytosine [C]) substitutes for another pyrimidine base or when a purine
    base (i.e., adenine [A] or guanine [G]) substitutes for another purine base.
    Transversion refers to a point mutation in DNA in which a single (two ring)
    purine (A or G) is changed for a (one ring) pyrimidine (T or C), or vice versa.

    Args:
      seq (str): original genomic sequence
      transition (bool): a flag indication if we want to perform transition
      transversion (bool): a flag indication if we want to perform transversion
      transition_prob (float): probability of transition being performed based on
      the mutation rate
      transversion_prob (float): probability of transversion being performed based
      on the mutation rate

    Returns:
      mutated_sequence (str): mutated sequence based on transitions and
      transversion
    """

    # create a list of random numbers with length of the input sequence
    transition_indexes = []
    transversion_indexes = []

    # mutation dict based on transitions
    transition_mutations = {"A": "G",
                            "G": "A",
                            "C": "T",
                            "T": "C"}

    # mutation dict based on transversions
    transversion_mutations = {"A": ["T", "C"],
                              "G": ["T", "C"],
                              "C": ["A", "G"],
                              "T": ["A", "G"]}

    # transistion mutations
    if transition:
        random_list = np.random.random(len(seq))
        transition_indexes = np.where(random_list <= transition_prob)[0]

    # transversion mutations
    if transversion:
        random_list = np.random.random(len(seq))
        transversion_indexes = np.where(random_list <= transversion_prob)[0]

    # enumerating the input sequence and perform the mutations
    mutated_seq = []
    for i, nucleotide in enumerate(seq):
        if i in transition_indexes:
            try:
                mutated_seq.append(transition_mutations[nucleotide])
            except KeyError:
                pass
        elif i in transversion_indexes:
            try:
                mutated_seq.append(transversion_mutations[nucleotide][round(np.random.uniform())])
            except KeyError:
                pass
        else:
            mutated_seq.append(nucleotide)

    return "".join(mutated_seq)


def reverse_complement(seq: str):
    seq = Seq(seq)
    rev_seq = str(seq.reverse_complement())
    return str(rev_seq)


def fragmentation(seq: str,
                  frag_len: int):
    start_index = np.random.randint(len(seq) - frag_len)
    return seq[start_index: start_index + frag_len]


def generate_pairs_mutation(data: pd.DataFrame,
                            mutation_rate: float,
                            class_to_idx,
                            k: int = 6,
                            number_of_pairs: int = 1):
    # initiate FCGR
    fcgr = CGR_utils.FCGR(k=k)

    # initialize the outputs (X_train, X_test, y_test)
    X_train = []
    X_test, y_test = [], []

    # iterate through the sequences
    for i, record in tqdm(data.iterrows(), total=data.shape[0]):
        label = class_to_idx[record.label]
        original_seq = record.sequence
        seq_length = len(original_seq)

        # DNA -> fcgr
        original_seq_fcgr = fcgr(original_seq)

        for i in range(number_of_pairs):
            mutated_seq = mutation(seq=original_seq,
                                   transition=True,
                                   transversion=True,
                                   transition_prob=mutation_rate,
                                   transversion_prob=(0.5 * mutation_rate))
            # DNA -> fcgr
            mutated_seq_fcgr = fcgr(mutated_seq)
            X_train.append((original_seq_fcgr / seq_length,
                            mutated_seq_fcgr / seq_length))

        X_test.append(original_seq_fcgr / seq_length)
        y_test.append(label)

    # datatype conversion
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float32")
    y_test = np.array(y_test)

    return X_train, X_test, y_test


def generate_pairs_fragmentation(data: pd.DataFrame,
                                 class_to_idx,
                                 frag_perc: float = 0.5,
                                 k: int = 6,
                                 number_of_pairs: int = 1):
    # initialize the outputs (X_train, X_test, y_test)
    X_train = []
    X_test, y_test = [], []

    # iterate through the sequences
    for i, record in tqdm(data.iterrows(), total=data.shape[0]):
        label = class_to_idx[record.label]
        original_seq = record.sequence
        seq_length = len(original_seq)

        # initiate FCGR
        fcgr = CGR_utils.FCGR(k=k)

        # DNA -> fcgr
        original_seq_fcgr = fcgr(original_seq)

        for i in range(number_of_pairs):
            augmented_seq = fragmentation(seq=original_seq,
                                          frag_len=int(frag_perc * seq_length))
            augmented_seq_fcgr = fcgr(augmented_seq)
            X_train.append((original_seq_fcgr / seq_length,
                            augmented_seq_fcgr / seq_length))

        X_test.append(original_seq_fcgr / seq_length)
        y_test.append(label)

    # datatype conversion
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float32")
    y_test = np.array(y_test)

    return X_train, X_test, y_test


def generate_pairs_mutation_twin(data: pd.DataFrame,
                                 mutation_rate_strong: float,
                                 mutation_rate_weak: float,
                                 class_to_idx,
                                 k: int = 6,
                                 number_of_pairs: int = 1):
    # initiate FCGR
    fcgr = CGR_utils.FCGR(k=k)

    # initialize the outputs (X_train, X_test, y_test)
    X_train = []
    X_test, y_test = [], []

    # iterate through the sequences
    for i, record in tqdm(data.iterrows(), total=data.shape[0]):
        label = class_to_idx[record.label]
        original_seq = record.sequence
        seq_length = len(original_seq)

        # DNA -> fcgr
        original_seq_fcgr = fcgr(original_seq)

        for i in range(number_of_pairs):
            mutated_seq_weak = mutation(seq=original_seq,
                                        transition=True,
                                        transversion=True,
                                        transition_prob=mutation_rate_weak,
                                        transversion_prob=(0.5 * mutation_rate_weak))
            mutated_seq_strong = mutation(seq=original_seq,
                                          transition=True,
                                          transversion=True,
                                          transition_prob=mutation_rate_strong,
                                          transversion_prob=(0.5 * mutation_rate_strong))
            # DNA -> fcgr
            mutated_seq_weak_fcgr = fcgr(mutated_seq_weak)
            mutated_seq_strong_fcgr = fcgr(mutated_seq_strong)
            X_train.append((mutated_seq_weak_fcgr / seq_length,
                            mutated_seq_strong_fcgr / seq_length))

        X_test.append(original_seq_fcgr / seq_length)
        y_test.append(label)

    # datatype conversion
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float32")
    y_test = np.array(y_test)

    return X_train, X_test, y_test
