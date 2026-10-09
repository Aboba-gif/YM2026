"""План E06: последовательности регуляризации и парные сравнения."""
from dataclasses import asdict, dataclass, fields
import math
import json
import re
from numbers import Integral, Real


ALPHA_EXPONENTS = tuple(-8.0 + 0.5 * index for index in range(25))
POPULATION_LENGTH_HOURS = 0.24109057879930182
PRIMARY_TIMES_HOURS = tuple(index / 3 for index in range(1, 10))
FULL_TIMES_HOURS = tuple(index / 24 for index in range(1, 73))
PRIMARY_TICKS = tuple(range(7, 72, 8))
STATION_SD = (1.0, 1.5, 2.0, 1.0)
SOURCES = ("PG10", "EC04")
PENALTIES = ("L2", "H1")


def _float(value, name, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class ObservationSpec:
    """Пространственное ядро и временной способ наблюдения.

    Parameters
    ----------
    spatial_kind : {'gaussian', 'compact'}
        Форма нормированного пространственного ядра.
    width_km : float
        Положительное координатное стандартное отклонение ядра в километрах.
    temporal_kind : {'snapshot', 'average'}, optional
        Мгновенное значение или среднее по предшествующему окну.
    window_hours : float or None, optional
        Положительная длительность окна в часах для average; для snapshot —
        None.
    """

    spatial_kind: str
    width_km: float
    temporal_kind: str = "snapshot"
    window_hours: float | None = None

    def __post_init__(self):
        if self.spatial_kind not in ("gaussian", "compact"):
            raise ValueError("spatial_kind must be gaussian or compact")
        object.__setattr__(self, "width_km", _float(self.width_km, "width_km"))
        if self.temporal_kind == "snapshot":
            if self.window_hours is not None:
                raise ValueError("snapshot must not declare an averaging window")
        elif self.temporal_kind == "average":
            object.__setattr__(self, "window_hours", _float(self.window_hours, "window_hours"))
        else:
            raise ValueError("temporal_kind must be snapshot or average")


@dataclass(frozen=True)
class StreamSpec:
    """Идентификатор воспроизводимого потока случайных чисел.

    Parameters
    ----------
    seed : int
        Положительный базовый seed.
    version : int
        Положительный номер версии потока.
    """

    seed: int
    version: int

    def __post_init__(self):
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(self, "version", _integer(self.version, "version"))


@dataclass(frozen=True)
class NoiseSpec:
    """Гауссовы ошибки с независимыми блоками четырёх постов.

    Parameters
    ----------
    family : str
        Временное семейство: station_exponential или
        station_exponential_mixture.
    station_sd : tuple of float, length 4
        Положительные стандартные отклонения в мкг/м³ в порядке постов.
    lengths_hours : tuple of float, optional
        Четыре положительные длины экспоненциальной корреляции в часах; для
        смеси — пустой кортеж.
    fast_hours : float or None, optional
        Положительная быстрая длина смеси в часах; для экспоненты — None.
    slow_hours : float or None, optional
        Медленная длина смеси в часах, больше fast_hours; для экспоненты —
        None.
    fast_weight : float or None, optional
        Доля быстрой компоненты смеси строго между нулём и единицей; для
        экспоненты — None.
    """

    family: str
    station_sd: tuple
    lengths_hours: tuple = ()
    fast_hours: float | None = None
    slow_hours: float | None = None
    fast_weight: float | None = None

    def __post_init__(self):
        sd = tuple(_float(x, "station_sd") for x in self.station_sd)
        lengths = tuple(_float(x, "lengths_hours") for x in self.lengths_hours)
        if len(sd) != 4:
            raise ValueError("E06 requires four station SDs in fixed station order")
        object.__setattr__(self, "station_sd", sd)
        object.__setattr__(self, "lengths_hours", lengths)
        if self.family == "station_exponential":
            if len(lengths) != 4 or any(x is not None for x in
                                      (self.fast_hours,self.slow_hours,self.fast_weight)):
                raise ValueError("exponential noise requires four lengths and no mixture parameters")
        elif self.family == "station_exponential_mixture":
            fast = _float(self.fast_hours, "fast_hours")
            slow = _float(self.slow_hours, "slow_hours")
            weight = _float(self.fast_weight, "fast_weight")
            if lengths or not fast < slow or not weight < 1:
                raise ValueError("mixture needs fast<slow, weight in (0,1), and no single lengths")
            for name,value in (("fast_hours",fast),("slow_hours",slow),("fast_weight",weight)):
                object.__setattr__(self, name, value)
        else:
            raise ValueError("unregistered E06 noise family")


@dataclass(frozen=True)
class ReuseReference:
    """Ссылка на последовательность исходного опыта E05.

    Parameters
    ----------
    condition : {'main', 'temporal_average'}
        Исходное условие research_validation_v2.
    source : {'PG10', 'EC04'}
        Идентификатор источника.
    replicate : int
        Номер реализации от 1 до 4; temporal_average допускает только 1 и 2.
    penalty : {'L2', 'H1'}
        Семейство регуляризации.
    study : {'research_validation_v2'}, optional
        Идентификатор исходного исследования.
    """

    condition: str
    source: str
    replicate: int
    penalty: str
    study: str = "research_validation_v2"

    def __post_init__(self):
        if self.study != "research_validation_v2" or self.condition not in ("main", "temporal_average"):
            raise ValueError("Only the two predeclared frozen-v2 conditions can be reused")
        _source_arm(self.source, self.replicate, self.penalty)
        object.__setattr__(self, "replicate", int(self.replicate))
        if self.condition == "temporal_average" and self.replicate > 2:
            raise ValueError("v2 temporal_average contains only replicates 1 and 2")


def _source_arm(source, replicate, penalty):
    if source not in SOURCES or penalty not in PENALTIES:
        raise ValueError("E06 inverse scope is PG10/EC04 and L2/H1")
    if _integer(replicate, "replicate") > 4:
        raise ValueError("E06 inverse replicates are 1 through 4")


@dataclass(frozen=True)
class PathSpec:
    """Условия последовательности оценок по 25 значениям регуляризации.

    Parameters
    ----------
    condition : str
        Непустой ASCII-идентификатор условия из букв, цифр и подчёркиваний.
    source : {'PG10', 'EC04'}
        Идентификатор источника.
    replicate : int
        Номер реализации от 1 до 4.
    penalty : {'L2', 'H1'}
        Квадратичная норма регуляризации временного источника.
    true_h : ObservationSpec
        Порождающий оператор наблюдения.
    inverse_h : ObservationSpec
        Предполагаемый оператор восстановления.
    weight : str
        Рабочая ковариация: W03, W_mix_oracle, W_exp_population или
        W_exp_estimated.
    noise : NoiseSpec
        Порождающая ковариационная модель ошибок.
    stream : StreamSpec
        Идентификатор потока случайных чисел.
    reuse : ReuseReference or None, optional
        Ссылка на исходную оценку того же источника, реализации и штрафа.
    grid, truth_grid : {'G0'}, optional
        Сетки восстановления и прямого порождающего расчёта.
    nodes : int, optional
        Число узлов P1-базиса; в E06 равно 73.
    tau_hours : float, optional
        Масштаб производной в норме H1, равный 0.25 ч.
    alpha_exponents : tuple of float, optional
        Фиксированные 25 показателей десятичного множителя регуляризации от
        −8 до 4 с шагом 0.5.
    """

    condition: str
    source: str
    replicate: int
    penalty: str
    true_h: ObservationSpec
    inverse_h: ObservationSpec
    weight: str
    noise: NoiseSpec
    stream: StreamSpec
    reuse: ReuseReference | None = None
    grid: str = "G0"
    truth_grid: str = "G0"
    nodes: int = 73
    tau_hours: float = 0.25
    alpha_exponents: tuple = ALPHA_EXPONENTS

    def __post_init__(self):
        if not isinstance(self.condition,str) or not self.condition or not all(
                x.isascii() and (x.isalnum() or x == "_") for x in self.condition):
            raise ValueError("condition must be a nonempty ASCII identifier")
        _source_arm(self.source,self.replicate,self.penalty)
        object.__setattr__(self,"replicate",int(self.replicate))
        if not isinstance(self.true_h,ObservationSpec) or not isinstance(self.inverse_h,ObservationSpec):
            raise TypeError("true_h and inverse_h must be ObservationSpec")
        if not isinstance(self.noise,NoiseSpec) or not isinstance(self.stream,StreamSpec):
            raise TypeError("explicit NoiseSpec and StreamSpec required")
        if self.weight not in ("W03","W_mix_oracle","W_exp_population","W_exp_estimated"):
            raise ValueError("unregistered E06 weight")
        if self.grid != "G0" or self.truth_grid != "G0" or _integer(self.nodes,"nodes") != 73:
            raise ValueError("E06 inverse design is fixed to G0/G0 and 73 P1 nodes")
        object.__setattr__(self,"nodes",int(self.nodes))
        tau = _float(self.tau_hours,"tau_hours")
        exponents = tuple(_float(x,"alpha exponent",positive=False) for x in self.alpha_exponents)
        if tau != 0.25 or exponents != ALPHA_EXPONENTS:
            raise ValueError("E06 requires tau=.25 and the fixed 25-alpha exponent grid")
        object.__setattr__(self,"tau_hours",tau)
        object.__setattr__(self,"alpha_exponents",exponents)
        if self.reuse is not None:
            if not isinstance(self.reuse,ReuseReference):
                raise TypeError("reuse must be ReuseReference or None")
            if (self.reuse.source,self.reuse.replicate,self.reuse.penalty) != (
                    self.source,self.replicate,self.penalty):
                raise ValueError("reuse reference must identify the same source/replicate/penalty")

    @property
    def id(self):
        """Вернуть идентификатор последовательности оценок.

        Returns
        -------
        str
            Условие, источник, реализация и штраф в формате
            condition/source/rN/penalty.
        """

        return f"{self.condition}/{self.source}/r{self.replicate}/{self.penalty}"

    @property
    def standard_normal_key(self):
        """Вернуть базовый ключ стандартных нормальных чисел.

        Returns
        -------
        tuple of int
            Seed, версия потока и реализация. Назначение и индекс выборки
            добавляются при генерации.
        """
        return self.stream.seed,self.stream.version,self.replicate

    @property
    def calibration_design_key(self):
        """Вернуть ключ общей калибровочной выборки.

        Returns
        -------
        tuple
            Полные параметры шума, поток и реализация; источник и оператор
            наблюдения не входят в ключ.
        """

        return self.noise,self.stream,self.replicate

    @property
    def data_design_key(self):
        """Вернуть ключ порождающей модели наблюдений.

        Returns
        -------
        tuple
            Источник, реализация, true_h, шум, поток и truth_grid; inverse_h в
            ключ не входит.
        """
        return self.source,self.replicate,self.true_h,self.noise,self.stream,self.truth_grid


_ALLOWED_CHANGES = {
    "penalty": ("penalty",),
    "assumed_spatial_H": ("inverse_h",),
    # Оператор наблюдения меняется одновременно в генераторе и восстановлении.
    
    "matched_spatial_design": ("true_h","inverse_h"),
    "assumed_temporal_H": ("inverse_h",),
    "covariance_family": ("weight",),
    "covariance_estimation": ("weight",),
}


@dataclass(frozen=True)
class ContrastSpec:
    """Ориентированное парное сравнение двух последовательностей.

    Parameters
    ----------
    key : str
        Непустой идентификатор вида сравнения.
    factor : str
        Изменяемый фактор: penalty, assumed_spatial_H,
        matched_spatial_design, assumed_temporal_H, covariance_family
        или covariance_estimation.
    left_path : str
        Идентификатор исходной последовательности.
    right_path : str
        Идентификатор сравниваемой последовательности, отличный от left_path.
    """

    key: str
    factor: str
    left_path: str
    right_path: str

    def __post_init__(self):
        if self.factor not in _ALLOWED_CHANGES:
            raise ValueError("unregistered E06 contrast factor")
        if not all(isinstance(x,str) and x for x in (self.key,self.left_path,self.right_path)):
            raise ValueError("contrast key and path identifiers must be nonempty strings")
        if self.left_path == self.right_path:
            raise ValueError("contrast must compare distinct paths")

    @property
    def id(self):
        """Вернуть идентификатор ориентированного сравнения.

        Returns
        -------
        str
            Ключ и упорядоченная пара идентификаторов последовательностей.
        """

        return self.key + ":" + self.left_path + "->" + self.right_path


@dataclass(frozen=True)
class StudyDesign:
    """Полный фиксированный план последовательностей и сравнений E06.

    При создании проверяется полнота и соответствие всех научных условий
    зарегистрированному плану.

    Parameters
    ----------
    paths : iterable of PathSpec
        Полный набор последовательностей в заданном порядке.
    contrasts : iterable of ContrastSpec
        Полный набор разрешённых парных сравнений.
    population_length_hours : float, optional
        Фиксированная популяционная длина корреляции в часах.
    calibration_panels : int, optional
        Число калибровочных выборок; в E06 равно 32.
    """

    paths: tuple
    contrasts: tuple
    population_length_hours: float = POPULATION_LENGTH_HOURS
    calibration_panels: int = 32

    def __post_init__(self):
        object.__setattr__(self,"paths",tuple(self.paths))
        object.__setattr__(self,"contrasts",tuple(self.contrasts))
        length = _float(self.population_length_hours,"population_length_hours")
        if length != POPULATION_LENGTH_HOURS or _integer(self.calibration_panels,"calibration_panels") != 32:
            raise ValueError("Population length and 32-panel calibration are fixed in E06")
        object.__setattr__(self,"population_length_hours",length)
        object.__setattr__(self,"calibration_panels",32)
        validate_design(self)

    @property
    def new_paths(self):
        """Выбрать последовательности без повторного использования E05.

        Returns
        -------
        tuple of PathSpec
            Новые последовательности в исходном порядке.
        """

        return tuple(p for p in self.paths if p.reuse is None)

    @property
    def reused_paths(self):
        """Выбрать последовательности со ссылкой на оценки E05.

        Returns
        -------
        tuple of PathSpec
            Повторно используемые последовательности в исходном порядке.
        """

        return tuple(p for p in self.paths if p.reuse is not None)

    @property
    def counts(self):
        """Подсчитать последовательности, попытки и сравнения плана.

        Returns
        -------
        dict of str to int
            Новая запись числа новых и повторных последовательностей,
            соответствующих попыток и сравнений.
        """

        return dict(new_paths=len(self.new_paths),reused_paths=len(self.reused_paths),
                    new_attempts=sum(len(p.alpha_exponents) for p in self.new_paths),
                    reused_attempts=sum(len(p.alpha_exponents) for p in self.reused_paths),
                    contrasts=len(self.contrasts))


def _conditions():
    """Составить таблицу заданных условий E06."""
    g1 = ObservationSpec("gaussian",1.)
    average = ObservationSpec("gaussian",1.,"average",1/3)
    corr = NoiseSpec("station_exponential",STATION_SD,(.25,)*4)
    af,ag,ass = (math.exp(-(1/3)/x) for x in (1/12,POPULATION_LENGTH_HOURS,1.))
    mixture = NoiseSpec("station_exponential_mixture",STATION_SD,
                        fast_hours=1/12,slow_hours=1.,fast_weight=(ass-ag)/(ass-af))
    stream2,stream3 = StreamSpec(20260926,2),StreamSpec(20260926,3)
    rows = [("spatial_G1_matched",g1,g1,"W03",corr,stream2,4,"main")]
    for tag,kernel in (("G05",ObservationSpec("gaussian",.5)),
                       ("G2",ObservationSpec("gaussian",2.)),
                       ("C1",ObservationSpec("compact",1.))):
        rows.extend([(f"spatial_{tag}_matched",kernel,kernel,"W03",corr,stream2,4,None),
                     (f"spatial_{tag}_assumed_G1",kernel,g1,"W03",corr,stream2,4,None)])
    for tag,weight in (("mix_oracle","W_mix_oracle"),("exp_population","W_exp_population"),
                       ("exp_estimated","W_exp_estimated")):
        rows.append((f"covariance_{tag}",g1,g1,weight,mixture,stream3,4,None))
    rows.extend([("temporal_average_matched",average,average,"W03",corr,stream2,2,"temporal_average"),
                 ("temporal_average_assumed_snapshot",average,g1,"W03",corr,stream2,2,None)])
    return tuple(rows)


def _contrast_table():
    rows = []
    for tag in ("G05","G2","C1"):
        rows.extend([(f"assumed_spatial/{tag}","assumed_spatial_H",
                      f"spatial_{tag}_matched",f"spatial_{tag}_assumed_G1"),
                     (f"matched_spatial/{tag}","matched_spatial_design",
                      "spatial_G1_matched",f"spatial_{tag}_matched")])
    rows.extend([("assumed_temporal/average20","assumed_temporal_H",
                  "temporal_average_matched","temporal_average_assumed_snapshot"),
                 ("covariance/family","covariance_family",
                  "covariance_mix_oracle","covariance_exp_population"),
                 ("covariance/estimation","covariance_estimation",
                  "covariance_exp_population","covariance_exp_estimated")])
    return tuple(rows)


def build_design():
    """Создать заданный план E06 без чтения файлов.

    Returns
    -------
    StudyDesign
        176 последовательностей и 224 парных сравнения в детерминированном
        порядке.
    """
    paths = []
    for condition,true_h,inverse_h,weight,noise,stream,repeats,reuse_id in _conditions():
        for source in SOURCES:
            for replicate in range(1,repeats+1):
                for penalty in PENALTIES:
                    ref = ReuseReference(reuse_id,source,replicate,penalty) if reuse_id else None
                    paths.append(PathSpec(condition,source,replicate,penalty,true_h,inverse_h,
                                          weight,noise,stream,ref))
    contrasts = []
    for p in paths:
        if p.penalty == "L2":
            contrasts.append(ContrastSpec("penalty/"+p.condition,"penalty",p.id,
                f"{p.condition}/{p.source}/r{p.replicate}/H1"))
    for key,factor,left,right in _contrast_table():
        for p in paths:
            if p.condition == left:
                contrasts.append(ContrastSpec(key,factor,p.id,
                    f"{right}/{p.source}/r{p.replicate}/{p.penalty}"))
    return StudyDesign(tuple(paths),tuple(contrasts))


def validate_design(design):
    """Проверить полноту плана и разрешённые изменения в парных сравнениях.

    Parameters
    ----------
    design : StudyDesign
        План для сверки с зарегистрированными условиями E06.
    """
    if not isinstance(design,StudyDesign):
        raise TypeError("design must be StudyDesign")
    if not all(isinstance(p,PathSpec) for p in design.paths):
        raise TypeError("all paths must be PathSpec")
    by_id = {p.id:p for p in design.paths}
    if len(by_id) != len(design.paths):
        raise ValueError("duplicate path ID")
    expected_ids = set()
    for condition,true_h,inverse_h,weight,noise,stream,repeats,reuse_id in _conditions():
        for source in SOURCES:
            for replicate in range(1,repeats+1):
                for penalty in PENALTIES:
                    expected = f"{condition}/{source}/r{replicate}/{penalty}"
                    expected_ids.add(expected)
                    p = by_id.get(expected)
                    if p is None or (p.true_h,p.inverse_h,p.weight,p.noise,p.stream) != (
                            true_h,inverse_h,weight,noise,stream):
                        raise ValueError("path coverage or scientific condition differs from E06")
                    ref = ReuseReference(reuse_id,source,replicate,penalty) if reuse_id else None
                    if p.reuse != ref:
                        raise ValueError("wrong or unregistered reuse reference")
    if set(by_id) != expected_ids:
        raise ValueError("unexpected path outside the finite E06 design")
    if not all(isinstance(c,ContrastSpec) for c in design.contrasts):
        raise TypeError("all contrasts must be ContrastSpec")
    identifiers = [c.id for c in design.contrasts]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate contrast ID")
    science = tuple(f.name for f in fields(PathSpec) if f.name not in ("condition","reuse"))
    expected_pairs = {(f"penalty/{p.condition}","penalty",p.id,
                       f"{p.condition}/{p.source}/r{p.replicate}/H1")
                      for p in design.paths if p.penalty == "L2"}
    for key,factor,left,right in _contrast_table():
        expected_pairs.update((key,factor,p.id,f"{right}/{p.source}/r{p.replicate}/{p.penalty}")
                              for p in design.paths if p.condition == left)
    actual_pairs = set()
    for c in design.contrasts:
        if c.left_path not in by_id or c.right_path not in by_id:
            raise ValueError("contrast references an unknown path")
        left,right = by_id[c.left_path],by_id[c.right_path]
        changes = {name for name in science if getattr(left,name) != getattr(right,name)}
        if changes != set(_ALLOWED_CHANGES[c.factor]):
            raise ValueError("contrast changes unintended scientific factors")
        if c.factor == "matched_spatial_design" and (
                left.true_h != left.inverse_h or right.true_h != right.inverse_h):
            raise ValueError("matched design must change true/inverse H coherently")
        actual_pairs.add((c.key,c.factor,c.left_path,c.right_path))
    if actual_pairs != expected_pairs:
        raise ValueError("contrast coverage differs from the predeclared finite list")



DIRECT_SOURCES = ("PG10", "SB150", "EC04", "EC06", "NEW-J2", "NEW-S2")
DIRECT_OBSERVATIONS = ("G1_snapshot", "G05_snapshot", "G2_snapshot", "C1_snapshot", "G1_average20")
DIRECT_DOMAINS = {"D0": "G0", "D1": "E06_D1_direct"}
DIRECT_D1 = dict(bounds_km=[-18., 12., -10., 10.], spacing_km=.25, steps=336)
QUADRATURE_SPACINGS = (.25, .125, .0625)
DEFAULT_FIGURES = (
    dict(id="spatial", contrast_keys=["assumed_spatial/G05", "assumed_spatial/G2", "assumed_spatial/C1"]),
    dict(id="covariance", contrast_keys=["covariance/family", "covariance/estimation"]),
)


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _ordered_ids(value, catalogue, name):
    if (type(value) is not list or any(type(item) is not str or not item for item in value)
            or len(set(value)) != len(value) or any(item not in catalogue for item in value)):
        raise ValueError(f"{name}: unique registered IDs required")
    selected = set(value)
    if value != [item for item in catalogue if item in selected]:
        raise ValueError(f"{name}: canonical relative order required")
    return value


def _design_record(paths, contrasts):
    return _json_copy(dict(paths=[asdict(p) for p in paths], contrasts=[asdict(c) for c in contrasts],
        population_length_hours=POPULATION_LENGTH_HOURS, calibration_panels=32))


def design_from_record(record):
    """Восстановить канонические условия из сохранённой части плана.

    Parameters
    ----------
    record : dict
        Четыре поля JSON-плана с полными дескрипторами и их исходным порядком.

    Returns
    -------
    dict
        Кортежи PathSpec и ContrastSpec, длина корреляции и число калибровок.
    """
    if type(record) is not dict or set(record) != {
            "paths", "contrasts", "population_length_hours", "calibration_panels"}:
        raise ValueError("Exact design fields required")
    full = build_design()
    path_ids = [f"{p['condition']}/{p['source']}/r{p['replicate']}/{p['penalty']}" for p in record["paths"]]
    contrast_ids = [f"{c['key']}:{c['left_path']}->{c['right_path']}" for c in record["contrasts"]]
    _ordered_ids(path_ids, [p.id for p in full.paths], "paths")
    _ordered_ids(contrast_ids, [c.id for c in full.contrasts], "contrasts")
    paths = tuple(p for p in full.paths if p.id in set(path_ids))
    contrasts = tuple(c for c in full.contrasts if c.id in set(contrast_ids))
    if any(c.left_path not in path_ids or c.right_path not in path_ids for c in contrasts):
        raise ValueError("Contrast endpoints must belong to the selected paths")
    if json.dumps(record, sort_keys=True, allow_nan=False) != json.dumps(_design_record(paths, contrasts), sort_keys=True):
        raise ValueError("Canonical scientific descriptors changed")
    return dict(paths=paths, contrasts=contrasts,
        population_length_hours=POPULATION_LENGTH_HOURS, calibration_panels=32)


def _direct_grid(value, name):
    if type(value) is not dict or set(value) != {"bounds_km", "spacing_km", "steps"}:
        raise ValueError(f"{name}: exact grid fields required")
    bounds = value["bounds_km"]
    if type(bounds) is not list or len(bounds) != 4:
        raise ValueError(f"{name}: four rectangle bounds required")
    bounds = [_float(v, name, positive=False) for v in bounds]
    h = _float(value["spacing_km"], name)
    _integer(value["steps"], name)
    for span in (bounds[1]-bounds[0], bounds[3]-bounds[2]):
        if span/h < 2 or not math.isclose(span/h, round(span/h), rel_tol=0., abs_tol=1e-11):
            raise ValueError(f"{name}: aligned rectangle with interior nodes required")
    return _json_copy(value)


def resolve_direct_spec(baseline_spec, direct=None):
    """Разрешить области, сетки, профили и квадратуру прямой серии.

    Parameters
    ----------
    baseline_spec : dict
        Конфигурация E05; G0 всегда берётся из grids.
    direct : dict or None, optional
        Явный прямой план либо стандартный полный план.

    Returns
    -------
    dict
        Отдельная запись со всеми выбранными сетками, включая G0.
    """
    value = dict(source_ids=list(DIRECT_SOURCES), domains=dict(DIRECT_DOMAINS),
        grids={DIRECT_DOMAINS["D1"]: _json_copy(DIRECT_D1)},
        observation_ids=list(DIRECT_OBSERVATIONS), quadrature_spacings_km=list(QUADRATURE_SPACINGS))
    if direct is not None:
        if type(direct) is not dict or set(direct) != set(value):
            raise ValueError("Exact direct plan fields required")
        value = _json_copy(direct)
    _ordered_ids(value["source_ids"], DIRECT_SOURCES, "direct sources")
    _ordered_ids(value["observation_ids"], DIRECT_OBSERVATIONS, "direct observations")
    if not value["source_ids"] or not value["observation_ids"]:
        raise ValueError("Direct sources and observations cannot be empty")
    domains = value["domains"]
    if (type(domains) is not dict or list(domains) != list(DIRECT_DOMAINS)
            or domains["D0"] != "G0" or type(domains["D1"]) is not str
            or not domains["D1"] or domains["D1"] == "G0"):
        raise ValueError("D0=G0 and a distinct named D1 grid required")
    grids = value["grids"]
    if type(grids) is not dict or set(grids) not in ({domains["D1"]}, set(domains.values())):
        raise ValueError("Only the two declared direct grids are permitted")
    g0 = _direct_grid(baseline_spec["grids"]["G0"], "G0")
    if "G0" in grids and grids["G0"] != g0:
        raise ValueError("Direct G0 differs from the baseline grid")
    value["grids"] = {"G0": g0, domains["D1"]: _direct_grid(grids[domains["D1"]], domains["D1"])}
    spacings = value["quadrature_spacings_km"]
    if type(spacings) is not list or not spacings:
        raise ValueError("Nonempty quadrature spacing sequence required")
    widths = [_float(s, "quadrature spacing") for s in spacings]
    if widths != sorted(set(widths), reverse=True):
        raise ValueError("Positive distinct decreasing quadrature spacings required")
    for grid in value["grids"].values():
        for h in widths:
            _direct_grid(dict(grid, spacing_km=h), "quadrature grid")
    return value


def default_direct_spec(baseline_spec):
    """Вернуть полный прямой план с G0 из конфигурации E05.

    Parameters
    ----------
    baseline_spec : dict
        Конфигурация с grids.G0.

    Returns
    -------
    dict
        Стандартные профили, операторы, D1 и шаги квадратуры.
    """
    return resolve_direct_spec(baseline_spec)


def plan_counts(paths, contrasts, source_ids):
    """Подсчитать условия, попытки и группы заданного плана.

    Parameters
    ----------
    paths, contrasts : sequence
        Канонические условия и сравнения.
    source_ids : sequence of str
        Объединение профилей прямого и обратного расчётов.

    Returns
    -------
    dict
        Числа последовательностей, сравнений, попыток, групп и профилей.
    """
    new = [p for p in paths if p.reuse is None]
    reused = [p for p in paths if p.reuse is not None]
    return dict(paths=len(paths), contrasts=len(contrasts), new_paths=len(new), reused_paths=len(reused),
        new_fit_attempts=sum(len(p.alpha_exponents) for p in new),
        reused_fit_attempts=sum(len(p.alpha_exponents) for p in reused),
        inverse_groups=len({(p.source, p.replicate) for p in paths}), source_profiles=len(set(source_ids)))


def resolve_figures(contrasts, figures=None):
    """Выбрать рисунки по каноническим сравнениям.

    Parameters
    ----------
    contrasts : sequence of ContrastSpec
        Сравнения выбранного плана.
    figures : list of dict or None, optional
        Записи id/contrast_keys; None выбирает применимую часть стандартных рисунков.

    Returns
    -------
    list of dict
        Отдельная копия списка с уникальными именами и исходным порядком ключей.
    """
    keys = list(dict.fromkeys(c.key for c in contrasts))
    if figures is None:
        figures = [dict(f, contrast_keys=[k for k in f["contrast_keys"] if k in keys])
                   for f in DEFAULT_FIGURES if any(k in keys for k in f["contrast_keys"])]
    if type(figures) is not list:
        raise ValueError("Figure list required")
    seen = set()
    for figure in figures:
        if (type(figure) is not dict or set(figure) != {"id", "contrast_keys"}
                or type(figure["id"]) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", figure["id"])
                or figure["id"] in seen):
            raise ValueError("Unique named figures required")
        seen.add(figure["id"])
        _ordered_ids(figure["contrast_keys"], keys, "figure contrast keys")
        if not figure["contrast_keys"]:
            raise ValueError("Each figure must contain a declared contrast")
    return _json_copy(figures)


def resolve_plan(configuration, baseline_spec):
    """Составить план E06 по конфигурации и завершённому опыту E05.

    Parameters
    ----------
    configuration : dict
        Конфигурация E06 версии 3 или 4.
    baseline_spec : dict
        Исходная конфигурация E05 с условиями и сетками.

    Returns
    -------
    dict
        Условия, сравнения, необходимые группы E05, счётчики и прямой план.
    """
    if type(configuration) is not dict:
        raise ValueError("Configuration object required")
    full = build_design()
    selection = configuration.get("selection", {})
    if type(selection) is not dict or set(selection) - {"path_ids", "contrast_ids"}:
        raise ValueError("Only path_ids and contrast_ids are allowed in selection")
    ids = _ordered_ids(selection.get("path_ids", [p.id for p in full.paths]),
                       [p.id for p in full.paths], "path_ids")
    paths = tuple(p for p in full.paths if p.id in set(ids))
    pairs = [c.id for c in full.contrasts if c.left_path in ids and c.right_path in ids]
    contrast_ids = _ordered_ids(selection.get("contrast_ids", pairs), [c.id for c in full.contrasts], "contrast_ids")
    contrasts = tuple(c for c in full.contrasts if c.id in set(contrast_ids))
    record = _design_record(paths, contrasts)
    design_from_record(record)
    expected_paths, baseline_groups = {}, []
    for p in paths:
        expected_paths.setdefault(f"{p.source}/r{p.replicate}", []).append(p.id)
        if p.source not in baseline_spec["sources"] or p.replicate not in baseline_spec["replicates"]:
            raise ValueError("Selected inverse group is absent from E05 configuration")
        if p.reuse is not None:
            group = (p.source, p.replicate)
            if group not in baseline_groups:
                baseline_groups.append(group)
            condition = next((c for c in baseline_spec["conditions"] if c["id"] == p.reuse.condition), None)
            expected = dict(weight=p.weight, grid=p.grid, truth_grid=p.truth_grid, nodes=p.nodes,
                tau_hours=p.tau_hours, availability="full", mask="none",
                temporal_H="average20" if p.true_h.temporal_kind == "average" else "snapshot",
                relocation_km=0., truth_gamma=baseline_spec["model"]["reaction_gamma"],
                inverse_gamma=baseline_spec["model"]["reaction_gamma"])
            if (condition is None or p.source not in condition["sources"] or p.replicate not in condition["replicates"]
                    or p.penalty not in condition["penalties"] or any(condition.get(k,
                        baseline_spec["truth_grid"] if k == "truth_grid" else None) != v for k, v in expected.items())):
                raise ValueError("Referenced E05 condition differs from the selected reuse path")
            noise = baseline_spec["noise"][condition["noise"]]
            if noise != dict(type="station_exponential", station_sd=list(p.noise.station_sd), ell_hours=list(p.noise.lengths_hours)):
                raise ValueError("Referenced E05 noise differs from the reuse path")
    declared_direct = configuration.get("direct")
    if "direct" in configuration and type(declared_direct) is not dict:
        raise ValueError("Explicit direct plan must be an object")
    if declared_direct is not None:
        if type(declared_direct.get("grids")) is not dict:
            raise ValueError("Direct grids must be an object")
        if "G0" in declared_direct["grids"]:
            raise ValueError("Configure G0 in the baseline, not in direct.grids")
    direct = resolve_direct_spec(baseline_spec, declared_direct)
    sources = [s for s in DIRECT_SOURCES if s in direct["source_ids"] or any(p.source == s for p in paths)]
    analysis = configuration.get("analysis", {})
    if type(analysis) is not dict or set(analysis) - {"figures"}:
        raise ValueError("Only figures are allowed in analysis")
    if "figures" in analysis and type(analysis["figures"]) is not list:
        raise ValueError("Explicit figure plan must be a list")
    figures = resolve_figures(contrasts, analysis.get("figures"))
    return dict(paths=paths, contrasts=contrasts, design=record, expected_paths=expected_paths,
        baseline_groups=tuple(baseline_groups), source_ids=tuple(sources),
        counts=plan_counts(paths, contrasts, sources), direct=direct, figures=_json_copy(figures))
