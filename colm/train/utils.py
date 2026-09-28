import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


def collate_fn(batch_dicts, pad_token_id):
    """Re-collate single-example batches (as sliced from a collated batch) into one batch.

    Each example keeps the right padding it received in its original large batch; the
    new batch is right-padded again to the longest example. Right padding plus the
    causal mask means padded positions never influence the loss.
    """
    if isinstance(batch_dicts, dict):
        batch_dicts = [batch_dicts]

    padding_values = {"labels": IGNORE_INDEX, "attention_mask": 0}
    collated = {}
    for key in batch_dicts[0]:
        values = [
            example[key][0]
            if isinstance(example[key], list) and len(example[key]) == 1
            else example[key]
            for example in batch_dicts
        ]
        if isinstance(values[0], torch.Tensor):
            values = [v.squeeze(0) if v.ndim == 2 and v.shape[0] == 1 else v for v in values]
            collated[key] = pad_sequence(
                values, batch_first=True, padding_value=padding_values.get(key, pad_token_id)
            )
        else:
            collated[key] = values
    return collated


def convert_to_ordered_range(arr):
    """Map arbitrary labels to 0..C-1 in sorted order."""
    value_to_new_value = {val: idx for idx, val in enumerate(np.unique(arr))}
    return np.array([value_to_new_value[val] for val in arr])


def increase_array_to_threshold(arr, threshold):
    arr = np.array(arr)
    difference = threshold - np.sum(arr)
    if difference < 0:
        raise ValueError(
            "The threshold must be greater than or equal to the current sum of the arr."
        )
    sorted_indices = np.argsort(arr)
    for i in range(difference):
        arr[sorted_indices[i % len(arr)]] += 1
    return arr


def increase_array_to_threshold_v2(min_arr, max_arr, threshold):
    min_arr = np.array(min_arr)
    max_arr = np.array(max_arr)
    difference = threshold - np.sum(min_arr)
    if difference < 0:
        raise ValueError(
            "The threshold must be greater than or equal to the current sum of the arr."
        )
    diff_indices = np.where(min_arr != max_arr)[0]
    shuffled_indices = np.random.permutation(diff_indices)
    for i in range(difference):
        min_arr[shuffled_indices[i % len(min_arr)]] += 1
    return min_arr


def decrease_array_to_threshold(arr, threshold):
    arr = np.array(arr)
    difference = np.sum(arr) - threshold
    if difference < 0:
        raise ValueError(
            "The threshold must be less than or equal to the current sum of the array."
        )
    sorted_indices = np.argsort(arr)[::-1]
    for i in range(difference):
        if arr[sorted_indices[i % len(arr)]] > 0:
            arr[sorted_indices[i % len(arr)]] -= 1
        else:
            raise ValueError("Cannot decrease elements further without making them negative.")
    return arr
