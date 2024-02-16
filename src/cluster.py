from pathlib import Path
import pandas as pd
import pickle
import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from utils import data_setup, model, engine, utils, loss_function

if __name__ == "__main__":
    # setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Hyperparameters
    DATASET = "1_Cypriniformes"  # Choos a dataset from [1_Cypriniformes, 2_Cyprinoidei, 3_Cyprinidae, 4_Cyprininae]
    K = 6
    RANDOM_SEED = 0
    BATCH_SIZE = 512
    NORMALIZATION_METHOD = "Min-Max"  # Min-Max or Frobenius

    EPOCHS = 1
    EMBEDDING_DIM = 512
    FEATURE_DIM = 128
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    TEMP_INS = 0.5
    TEMP_CLU = 0.1

    # Create data directory
    DATA_PATH = Path("data/" + DATASET)
    DF_NAME = "balanced_data.pkl"
    TRAIN_FILE = "train_data.pkl"
    DF_SAVE_PATH = DATA_PATH / DF_NAME

    records_df = pd.read_pickle(DF_SAVE_PATH)
    class_names = sorted(records_df.label.unique())
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    with open(DATA_PATH / TRAIN_FILE, "rb") as handle:
        X_train, X_test, y_test = pickle.load(handle)

    # data normalization
    X_train, X_test = utils.data_normalization(X_train, X_test, method=NORMALIZATION_METHOD)
    print(f"Class names: {class_names}")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape} | Number of labels in y_test: {len(y_test)}")
    ####################################################################################################################
    # Create Datasets and DataLoaders
    random.seed(RANDOM_SEED)
    NUM_WORKERS = 1

    train_data = data_setup.PairSeqData(train_pairs=X_train,
                                        transform=None)
    test_data = data_setup.SeqData(sequences=X_test,
                                   labels=y_test,
                                   classes=class_names,
                                   class_to_idx=class_to_idx,
                                   transform=None)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  drop_last=False,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=len(test_data),
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
    ####################################################################################################################
    FEATURE_DIMs = [32, 64, 128, 256, 512]
    LRs = [1e-5, 5e-5, 8e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    TEMP_INSs = np.arange(0, 1, 0.05)
    TEMP_CLUs = np.arange(0, 1, 0.05)

    for FEATURE_DIM in FEATURE_DIMs:
        for LR in LRs:
            for TEMP_INS in TEMP_INSs:
                for TEMP_CLU in TEMP_CLUs:
                    print("#" * 100)
                    print(f"Feature dim: {FEATURE_DIM} | lr: {LR} | temp ins: {TEMP_INS} | temp clu {TEMP_CLU}")
                    # Training
                    torch.manual_seed(RANDOM_SEED)
                    torch.cuda.manual_seed(RANDOM_SEED)
                    # initialize the model
                    # backbone_model = model.BackBoneModel(input_shape=1,
                    #                                      output_shape=EMBEDDING_DIM)
                    backbone_model = model.get_resnet("ResNet18")
                    projector_model = model.Network(backbone=backbone_model,
                                          rep_dim=EMBEDDING_DIM,
                                          feature_dim=FEATURE_DIM,
                                          class_num=len(class_names)).to(device)
                    # Setup loss function and optimizer
                    optimizer = torch.optim.Adam(
                        [
                            {"params": projector_model.backbone.parameters(), "lr": LR, },
                            {"params": projector_model.instance_projector.parameters(), "lr": LR},
                            {"params": projector_model.cluster_projector.parameters(), "lr": LR},
                        ],
                        lr=LR,
                        weight_decay=WEIGHT_DECAY,
                    )
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
                    criterion_instance = loss_function.InstanceLoss(BATCH_SIZE, TEMP_INS, device).to(device)
                    criterion_cluster = loss_function.ClusterLoss(len(class_names), TEMP_CLU, device).to(device)

                    # start the timer
                    start_time = timer()

                    # train model
                    model_results = engine.train(model=projector_model,
                                                 train_dataloader=train_dataloader,
                                                 test_dataloader=test_dataloader,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 criterion_instance=criterion_instance,
                                                 criterion_cluster=criterion_cluster,
                                                 epochs=EPOCHS)

                    # end the timer
                    end_time = timer()
                    total_time = (end_time - start_time)
                    print(f"Total training time: {total_time:.3f} seconds")

                    print(f"Evaluating model")
                    y_pred, ind, acc = engine.model_evaluation(model=projector_model, X_test=X_test, y_test=y_test)
                    print(f"Accuracy of model: {acc * 100:.2f}%")
                    # utils.plot_loss_curves(model_results, total_time=total_time)
                    print("#" * 100)


