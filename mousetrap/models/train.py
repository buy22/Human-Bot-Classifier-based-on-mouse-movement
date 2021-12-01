"""
train.py - training loop function for PyTorch models.
"""

from typing import Callable, Dict, Tuple

import torch
import torch.utils.data
from tqdm import tqdm

def train(
    model: torch.nn.Module, 
    epochs: int, 
    criterion: Callable,
    optimizer: Callable,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader = None,
    device: str = 'cpu') -> Tuple[torch.nn.Module, Dict]:
    """
    Trains the given model, for a certain number of epochs, based on the loss
    function defined by `criterion`. 

    A train dataloader must also be provided, and optionally a validation
    dataloader as well. These should be of type `torch.utils.data.Dataloader`.

    Args:
        model: torch.nn.Module
            The model to train.
        epochs: int
            Number of epochs to train for.
        criterion: Callable
            The loss function to use for training.
        optimizer: Callable
            The optimizer to use for training.
        train_dataloader: torch.utils.data.DataLoader
            The dataloader for the training data.
        val_dataloader: torch.utils.data.DataLoader, optional
            The dataloader for the validation data.
        device: str, optional.
            The device to perform training on. Default 'cpu'. 
    Returns:
        torch.nn.Module
            The trained model.
    """
    # TODO: fix this
    # temporary hack to ensure that batch size is 1
    assert train_dataloader.batch_size == 1
    if val_dataloader:
        assert val_dataloader.batch_size == 1

    # send the model to device
    model = model.to(device)

    # store statistics across epochs
    best_loss = float('inf')
    best_acc = 0.0
    best_recall = 0.0
    best_precision = 0.0

    # begin training
    for epoch in range(1, epochs + 1):
        # print and keep some statistics
        print(f'Epoch {epoch}: ')
        running_loss = 0.0
        running_acc = 0.0
        prec_denom = 0
        prec_num = 0
        recall_denom = 0
        recall_num = 0

        # run through the training dataset
        model.train()
        for datum in tqdm(train_dataloader):
            x = datum['data'][0].to(device)
            y = torch.Tensor([datum['label']]).to(device).unsqueeze(0)

            # zero the optimizer gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # find the accuracy
            preds = (outputs > 0.5)
            acc = (preds == y).sum().item() / x.size(0)

            # precision and recall
            # assumes that the batch size is 1. precision and recall
            # are calculated from the perspective of bot detection.
            if (y.sum().item() == 1):
                recall_denom += 1
                if (preds.sum().item() == 1):
                    recall_num += 1
            if (preds.sum().item() == 1):
                prec_denom += 1
                if (y.sum().item() == 1):
                    prec_num += 1

            # update statistics
            running_loss += loss.item()
            running_acc += acc
        
        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        print("[%d] loss: %.3f accuracy: %.3f" % (epoch, running_loss, running_acc))

        # precision and recall
        prec = prec_num * 1.0 / prec_denom if prec_denom > 0 else 0
        recall = recall_num * 1.0 / recall_denom if recall_denom > 0 else 0

        # save stats
        if running_acc > best_acc:
            best_acc = running_acc
        if running_loss < best_loss:
            best_loss = running_loss
        if prec > best_precision:
            best_precision = prec
        if recall > best_recall:
            best_recall = recall

        # every few epochs, run through the validation set
        # TODO

    # save stats in a dictionary
    # print(prec_denom)
    # print(recall_denom)
    stats = {
        'loss': best_loss,
        'accuracy': best_acc,
        'recall': best_recall,
        'precision': best_precision
    }
    return model.to('cpu'), stats