from pathlib import Path
import pandas as pd
import pickle
import torch
import numpy as np
from scipy import stats
import os
import argparse, sys
import random
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from utils import data_setup, model, engine, utils, loss_function

if __name__ == "__main__":
    # setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="1_Cypriniformes", type=str,
        help="choose a dataset from 1_Cypriniformes, 2_Cyprinoidei, 3_Cyprinidae, 4_Cyprininae"
    )
    parser.add_argument("--k", default=6, type=int,
                        help="k-mer size, an integer between 6-8")
    parser.add_argument("--pairs_file_name", default="train_data_frag_0.95_0.8.pkl", type=str,
                        help="name of training data file")
    parser.add_argument("--number_of_models", default=5, type=int,
                        help="number of models")
    parser.add_argument("--lr", default=3e-4, type=float,
                        help="learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="weight decay")
    parser.add_argument("--temp_ins", default=0.15, type=float,
                        help="instance temperature")
    parser.add_argument("--temp_clu", default=0.9, type=float,
                        help="cluster temperature")
    parser.add_argument("--num_epochs", default=80, type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch size")
    parser.add_argument("--embedding_dim", default=512, type=int,
                        help="embedding dimension")
    parser.add_argument("--feature_dim", default=128, type=int,
                        help="feature dimension")
    args = parser.parse_args()
    print(args)

    # Hyperparameters
    RANDOM_SEED = 0
    NORMALIZATION_METHOD = "Frobenius"  # Min-Max or Frobenius

    # Create data directory
    DATA_PATH = Path("data/" + args.dataset)
    DF_NAME = "balanced_data.pkl"
    DF_SAVE_PATH = DATA_PATH / DF_NAME

    records_df = pd.read_pickle(DF_SAVE_PATH)
    class_names = sorted(records_df.label.unique())
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    with open(DATA_PATH / args.pairs_file_name, "rb") as handle:
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
                                  batch_size=args.batch_size,
                                  num_workers=NUM_WORKERS,
                                  drop_last=False,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=len(test_data),
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)
    ####################################################################################################################
    y_preds = []
    for i in range(args.number_of_models):
        # Training
        print(f"Training model #{i + 1}")
        torch.manual_seed(RANDOM_SEED + i)
        torch.cuda.manual_seed(RANDOM_SEED + i)
        # initialize the model
        backbone_model = model.BackBoneModel(input_shape=1,
                                             output_shape=args.embedding_dim)
        # backbone_model = model.get_resnet("ResNet18")
        projector_model = model.Network(backbone=backbone_model,
                                        rep_dim=args.embedding_dim,
                                        feature_dim=args.feature_dim,
                                        class_num=len(class_names)).to(device)
        # Setup loss function and optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": projector_model.backbone.parameters(), "lr": args.lr, },
                {"params": projector_model.instance_projector.parameters(), "lr": args.lr},
                {"params": projector_model.cluster_projector.parameters(), "lr": args.lr},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        criterion_instance = loss_function.InstanceLoss(args.batch_size, args.temp_ins, device).to(device)
        criterion_cluster = loss_function.ClusterLoss(len(class_names), args.temp_clu, device).to(device)

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
                                     epochs=args.num_epochs)

        # end the timer
        end_time = timer()
        total_time = (end_time - start_time)
        print(f"Total training time: {total_time:.3f} seconds")

        print(f"Evaluating model")
        y_pred, ind, acc = engine.model_evaluation(model=projector_model, X_test=X_test, y_test=y_test)
        print(f"Accuracy of model: {acc * 100:.2f}%")
        # utils.plot_loss_curves(model_results, total_time=total_time)
        d = {}
        for j, k in ind:
            d[j] = k
        for j in range(len(y_pred)):  # we do this for each sample or sample batch
            y_pred[j] = d[y_pred[j]]
        y_preds.append(y_pred)
        print("#" * 100)

    ####################################################################################################################
    # Majority voting
    y_preds = np.array(y_preds)
    mode, counts = stats.mode(y_preds, axis=0)

    w = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for i in range(y_test.shape[0]):
        w[y_test[i], mode[i]] += 1
    print(f"Accuracy of ensemble of {args.number_of_models} models: {np.sum(np.diag(w) / np.sum(w))}")
    print("Confusion matrix:")
    print(w)
