"""Проверки неотрицательной добавки и отделения подгонки от последних точек."""

import numpy as np


from experiments.kraz_alignment.run import choose_templates as choose


def test_template_choice_matches_direct_constrained_offset_minimization():
    templates = np.array([[0., 1., 2., 1., 0., 0., 0., 0., 0.],
                          [0., 3., 6., 3., 0., 0., 0., 0., 0.]])
    values = np.array([[2., 5., 8., 5., 2., 2., 3., 4., 5.],
                       [0., .5, 1., .5, 0., 0., 3., 4., 5.]])
    choices, offsets, predictions, scores = choose(values, templates, 6)
    np.testing.assert_array_equal(choices, [1, 0])
    np.testing.assert_allclose(offsets, [2., 0.])
    for row in range(len(values)):
        # Каждый допустимый множитель и соответствующая неотрицательная постоянная добавка
        # проверяются независимым скалярным расчётом наименьших квадратов, включая активную границу.
        candidates = []
        for template in templates:
            intercept = max(0., sum(values[row, k]-template[k] for k in range(6))/6)
            candidates.append(sum((values[row,k]-template[k]-intercept)**2 for k in range(6)))
        denominator = sum((values[row,k]-np.mean(values[row,:6]))**2 for k in range(6))
        assert np.isclose(scores[row], min(candidates)/denominator)


def test_withheld_values_cannot_change_choice_offset_or_window_score():
    templates = np.array([np.linspace(0, 10, 9), np.linspace(5, 0, 9)])
    values = np.array([templates[0]+2, templates[1]+4])
    before = choose(values, templates, 6)
    values[:, 6:] = [[1e8, 0, 50], [100, 1e6, 3]]
    after = choose(values, templates, 6)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)
