"""Семейство методов заполнения PM₂.₅ с общим состоянием обучения."""
from dataclasses import dataclass

import numpy as np
from timeseries.contracts import MaskedInputs, result_from_inputs
from timeseries.imputation import ImputerFamily, build_imputer, input_layout

from .models import CoreModels
from .data import SITES


@dataclass(frozen=True)
class PM25Fit:
    """Хранить совместно обученные модели одного целевого поста.

    Parameters
    ----------
    models : CoreModels
        Состояние четырёх кандидатов после обучения.
    target_site : str
        Целевой пост из ``SITES``.
    layout : str
        JSON сетки, осей и единиц из ``input_layout``.
    """
    models: CoreModels
    target_site: str
    layout: str


def _parameters(parameters):
    expected = {"config", "target_site", "method", "gap"}
    if set(parameters) != expected:
        raise ValueError("PM25 parameters require config, target_site, method and gap")
    if parameters["target_site"] not in SITES:
        raise ValueError("Unknown target station")
    if parameters["method"] not in {"linear", "common_hgb", "network_hgb", "climatology"}:
        raise ValueError("Unknown PM25 method")
    gap = parameters["gap"]
    if gap is not None and (not isinstance(gap, list) or len(gap) != 2
                            or any(type(i) is not int for i in gap) or not 0 <= gap[0] < gap[1]):
        raise ValueError("Gap must be null or two ordered nonnegative integer bounds")
    return {**parameters, "config": CoreModels(parameters["config"]).config}


@dataclass(frozen=True)
class PM25Imputer:
    """Обучать кандидатов совместно и возвращать результат выбранного метода.

    Parameters
    ----------
    config : dict
        Проверенная конфигурация ``CoreModels``.
    target_site : str
        Пост из ``SITES``.
    method : {'linear', 'common_hgb', 'network_hgb', 'climatology'}
        Метод предсказания блока.
    gap : list of int or None
        Полуоткрытые границы блока; None допускается при сборке для обучения.
    """
    config: dict
    target_site: str
    method: str
    gap: list | None

    def fit(self, train_inputs, *, rng, train_mask):
        """Обучить четыре кандидата по видимым исходным обучающим меткам.

        Parameters
        ----------
        train_inputs : MaskedInputs
            Входы выборки ``train``; значения вне ``train_mask`` скрыты.
        rng : numpy.random.Generator
            Генератор вызывающей процедуры. HGB использует ``random_state``
            из конфигурации, поэтому этот метод не расходует ``rng``.
        train_mask : ndarray of bool, shape (n,)
            Допустимые обучающие отсчёты.

        Returns
        -------
        PM25Fit
            Одно общее состояние четырёх кандидатов целевого поста.
        """
        if not isinstance(train_inputs, MaskedInputs) or train_inputs.population_kind != "train":
            raise ValueError("Fit requires original training inputs")
        train_mask = np.asarray(train_mask)
        if train_mask.dtype != np.bool_ or train_mask.shape != (len(train_inputs.times),):
            raise ValueError("train_mask must be a Boolean array aligned with index")
        if train_inputs.visible[~train_mask].any():
            raise ValueError("Values outside train_mask must be hidden")
        models = CoreModels(self.config).fit(train_inputs, train_mask, SITES.index(self.target_site))
        return PM25Fit(models, self.target_site, input_layout(train_inputs))

    def predict(self, inputs, *, fitted):
        """Предсказать один блок, сохранив все видимые исходные значения.

        Parameters
        ----------
        inputs : MaskedInputs
            Текущие видимые входы с полностью скрытым целевым блоком.
        fitted : PM25Fit
            Общее состояние того же поста, конфигурации и сетки.

        Returns
        -------
        ImputationResult
            Прогноз только целевого блока. Неприменимый метод оставляет
            блок недоступным; дополнительный скрытый контекст остаётся NaN.
            Оценка неопределённости не задаётся.
        """
        if not isinstance(inputs, MaskedInputs) or not isinstance(fitted, PM25Fit):
            raise ValueError("Prediction requires MaskedInputs and PM25Fit")
        if (fitted.target_site != self.target_site or fitted.models.config != self.config
                or fitted.layout != input_layout(inputs)):
            raise ValueError("Training and prediction configurations or layouts differ")
        if self.gap is None:
            raise ValueError("Prediction requires explicit gap bounds")
        start, end = self.gap
        target = SITES.index(self.target_site)
        predicted = fitted.models.predict_gap(inputs, start, end, target, method=self.method)
        values = inputs.values.copy()
        if predicted is not None:
            values[start:end, target, 0] = predicted
        return result_from_inputs(inputs, values)


PM25_FAMILY = ImputerFamily(_parameters, lambda p: PM25Imputer(**p))


def pm25_imputer(config, target_site, *, method="climatology", gap=None):
    """Собрать метод PM₂.₅ через публичный конструктор семейства.

    Parameters
    ----------
    config : dict
        Настройки ``CoreModels``.
    target_site : str
        Целевой пост из ``SITES``.
    method : str, optional
        Один из четырёх кандидатов, по умолчанию ``climatology``.
    gap : list of int or None, optional
        Полуоткрытые границы предсказываемого блока.

    Returns
    -------
    PM25Imputer
        Проверенный объект метода без вычисления прогноза или обучения.
    """
    return build_imputer({"kind": "pm25", "parameters": {
        "config": config, "target_site": target_site, "method": method, "gap": gap}},
        families={"pm25": PM25_FAMILY})


def prediction_results(inputs, fitted, start, end):
    """Вычислить четыре связанных прогноза одного блока.

    Parameters
    ----------
    inputs : MaskedInputs
        Видимые входы с полностью скрытым целевым блоком.
    fitted : PM25Fit
        Общее обученное состояние целевого поста.
    start, end : int
        Локальные полуоткрытые границы блока.

    Returns
    -------
    dict of str to ImputationResult
        Результаты в порядке linear, common_hgb, network_hgb, climatology.
    """
    results = {}
    for method in ("network_hgb", "climatology", "linear", "common_hgb"):
        results[method] = pm25_imputer(fitted.models.config, fitted.target_site,
                                       method=method, gap=[start, end]).predict(inputs, fitted=fitted)
    return {method: results[method] for method in ("linear", "common_hgb", "network_hgb", "climatology")}
