"""Проверки численных входов и независимости возвращённой копии."""
import numpy as np
import pytest

from experiments.source_comparison.arrays import finite_real_array


@pytest.mark.parametrize("values, expected", [
    ([1, -2], [1., -2.]),
    ([True, False], [1., 0.]),
    (["1.25", "-2.5"], [1.25, -2.5]),
    (np.array([1.25, -2.5], dtype=object), [1.25, -2.5]),
    (np.empty((0, 2)), np.empty((0, 2))),
    (np.array([[1., 2.], [3., 4.]], dtype=np.float32), [[1., 2.], [3., 4.]]),
    (True, 1.),
])
def test_numeric_conversion_preserves_values_and_shape(values, expected):
    result = finite_real_array(values, "input")
    np.testing.assert_array_equal(result, expected)
    assert result.shape == np.asarray(values).shape
    assert result.dtype == np.dtype(float)


def test_read_only_strided_input_and_output_remain_independent():
    original = np.arange(8., dtype=float)
    values = original[::-2]
    values.flags.writeable = False
    result = finite_real_array(values, "input")
    np.testing.assert_array_equal(result, [7., 5., 3., 1.])
    assert result.flags.writeable and not np.shares_memory(result, values)
    result[0] = 99.
    assert original[-1] == 7.
    original[-3] = 88.
    assert result[1] == 5.


@pytest.mark.parametrize("values", [[1.+0j], [1.+2j], [np.nan], [np.inf], ["bad"]])
def test_invalid_real_values_are_rejected(values):
    with pytest.raises(ValueError):
        finite_real_array(values, "input")


@pytest.mark.parametrize("values", [np.array([1.+0j], dtype=object), [{}]])
def test_unsupported_conversion_keeps_numpy_type_error(values):
    with pytest.raises(TypeError):
        finite_real_array(values, "input")
