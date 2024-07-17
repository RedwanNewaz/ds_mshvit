import numpy as np
import torch


def _get_unique_count(labels):
    if labels.ndim == 1:
        unique_labels, unique_counts = np.unique(labels, return_counts = True)
    elif labels.ndim == 2:
        unique_counts = [len(labels[labels[:, idx] == 1]) for idx in range(labels.shape[-1])]
        unique_labels = [idx for idx in range(labels.shape[-1])]
    
    return unique_labels, unique_counts

def identity_weight(labels, num_classes):
    class_weights = np.zeros(num_classes)
    for label_idx in range(num_classes):
        class_weights[label_idx] = 1.0

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()
    
def effective_samples(labels, num_classes, beta, rescale_classes = True):
    class_weights = np.zeros(num_classes)
    unique_labels, unique_counts = _get_unique_count(labels)
    
    for label_idx in unique_labels:
        effective_number = 1 - np.power(beta, unique_counts[label_idx])
        class_weights[label_idx] = (1 - beta) / effective_number

    class_weights = class_weights / np.sum(class_weights)

    if rescale_classes:
        class_weights *=  num_classes

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()