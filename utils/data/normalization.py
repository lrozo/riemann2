import torch


def standard_normalization(data):
    data_mean = data.mean(dim=0)
    centered_data = data - data_mean
    data_std = centered_data.std(dim=0)
    standardized_data = centered_data / data_std

    return standardized_data, data_mean, data_std


def minmax_normalization(data):
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]

    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data, data_min, data_max


def centering(data):
    data_mean = data.mean(dim=0)
    centered_data = data - data_mean

    return centered_data, data_mean

