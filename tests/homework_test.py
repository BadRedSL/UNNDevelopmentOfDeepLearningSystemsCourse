from vg_beat_detectors import FastNVG, FastWHVG
import pytest
import numpy as np
from scipy.datasets import electrocardiogram

triangle_signal = -abs(np.arange(1000) - 501)
ones_signal = np.ones(1000)


def test_triangle_signal_nvg():
    """Tests basic usage with a triangular shaped signal. Should detect the middle point. FastNVG algorithm."""
    detected = FastNVG(sampling_frequency=360).find_peaks(triangle_signal)
    print(detected)
    assert detected == [501]


def test_triangle_signal_whvg():
    """Tests basic usage with a triangular shaped signal. Should detect the middle point. FastWHVG algorithm."""
    detected = FastWHVG(sampling_frequency=360).find_peaks(triangle_signal)
    assert detected == [501]


def test_ones_nvg():
    """Tests basic NVG usage with a const signal. Should detect nothing."""
    detected = FastNVG(sampling_frequency=360).find_peaks(ones_signal)
    assert detected.size == 0


def test_ones_whvg():
    """Tests basic WHVG usage with a const signal. Should detect nothing."""
    detected = FastWHVG(sampling_frequency=360).find_peaks(ones_signal)
    assert detected.size == 0


def test_none_input_nvg():
    """Tests peak NVG detection with None input."""
    with pytest.raises(ValueError):
        FastNVG(sampling_frequency=360).find_peaks(None)


def test_none_input_whvg():
    """Tests peak WHVG detection with None input."""
    with pytest.raises(ValueError):
        FastWHVG(sampling_frequency=360).find_peaks(None)


# def test_str_input_nvg():
#     """Tests peak NVG detection with wrong input."""
#     signal = [str(i) for i in triangle_signal]
#     with pytest.raises(ValueError):
#         FastNVG(sampling_frequency=360).find_peaks(signal)
#
#
# def test_str_input_whvg():
#     """Tests peak WHVG detection with wrong input."""
#     signal = [str(i) for i in triangle_signal]
#     with pytest.raises(ValueError):
#         FastWHVG(sampling_frequency=360).find_peaks(signal)

def test_mse_nvg():
    """This nvg test calculates MSE between calculated peaks and their actual coordinates.
     It's should be less than 10."""

    def calculate_mse(predictions, targets):
        predictions = np.unique(predictions)
        return np.mean((targets - predictions) ** 2)

    ecg = electrocardiogram()[0:5700]
    detected = FastNVG(sampling_frequency=360).find_peaks(ecg)
    expected = np.array([125, 343, 552, 748, 944, 1130, 1317, 1501, 1691, 1880, 2065, 2251, 2431, 2608, 2779, 2956,
                         3125, 3292, 3456, 3614, 3776, 3948, 4129, 4310, 4482, 4652, 4812, 4984, 5157, 5323, 5496,
                         5674])
    assert calculate_mse(detected, expected) < 10


def test_mse_whvg():
    """This whvg test calculates MSE between calculated peaks and their actual coordinates. It's should be less than 10."""

    def calculate_mse(predictions, targets):
        predictions = np.unique(predictions)
        return np.mean((targets - predictions) ** 2)

    ecg = electrocardiogram()[0:5700]
    detected = FastWHVG(sampling_frequency=360).find_peaks(ecg)
    expected = np.array([125, 343, 552, 748, 944, 1130, 1317, 1501, 1691, 1880, 2065, 2251, 2431, 2608, 2779, 2956,
                         3125, 3292, 3456, 3614, 3776, 3948, 4129, 4310, 4482, 4652, 4812, 4984, 5157, 5323, 5496,
                         5674])
    assert calculate_mse(detected, expected) < 10
