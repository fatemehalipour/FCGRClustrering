from pathlib import Path
from typing import Dict
import pandas as pd

from utils import augmentation_utils


def generate_pairs(data: pd.DataFrame,
                   class_to_idx: Dict,
                   weak_mutation_rate: float,
                   strong_mutation_rate: float,
                   number_of_pairs: int = 1,
                   k: int = 6):
    """
    # TODO: Complete the comments
    Generate pairs of (weak augmentation, strong augmentation
    """
    X_train, X_test, y_test = augmentation_utils.generate_pairs_mutation_twin(data=data,
                                                                              mutation_rate_weak=weak_mutation_rate,
                                                                              mutation_rate_strong=strong_mutation_rate,
                                                                              class_to_idx=class_to_idx,
                                                                              k=k,
                                                                              number_of_pairs=number_of_pairs)
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape} | Number of labels in y_test: {len(y_test)}")
    return [X_train, X_test, y_test]


if __name__ == "__main__":
    # Hyperparameters
    K = 6
    WEAK_MUTATION_RATE = 1e-3
    STRONG_MUTATION_RATE = 1e-2
    NUMBER_OF_PAIRS = 1

    # Create data directory
    DATA_PATH = Path("../data/1_Cypriniformes")
    DF_NAME = "balanced_data.pkl"
    TRAIN_FILE = "train_data.pkl"
    DF_SAVE_PATH = DATA_PATH / DF_NAME
    TRAIN_DATA_PATH = DATA_PATH / TRAIN_FILE

    records_df = pd.read_pickle(DF_SAVE_PATH)
    class_names = sorted(records_df.label.unique())
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    data = generate_pairs(data=records_df,
                          weak_mutation_rate=WEAK_MUTATION_RATE,
                          strong_mutation_rate=STRONG_MUTATION_RATE,
                          class_to_idx=class_to_idx,
                          k=K,
                          number_of_pairs=NUMBER_OF_PAIRS)
