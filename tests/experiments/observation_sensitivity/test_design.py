"""Состав опыта E06, сравнения разрешённых факторов и допустимое повторное использование записей v2."""
from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace
import builtins
import json
import math
from pathlib import Path

import pytest

from experiments.observation_sensitivity.design import ALPHA_EXPONENTS, FULL_TIMES_HOURS, PRIMARY_TICKS, PRIMARY_TIMES_HOURS, ContrastSpec, NoiseSpec, ObservationSpec, ReuseReference, StreamSpec, StudyDesign, build_design, validate_design


# Независимая таблица протокола задаёт максимальный повтор, порождающее и предполагаемое ядра, пару
# временных типов, вес и условие повторного использования. Она не выводится из реализации.
EXPECTED = {
    "spatial_G1_matched": (4,("gaussian",1.),("gaussian",1.),"snapshot","snapshot","W03","main"),
    "spatial_G05_matched": (4,("gaussian",.5),("gaussian",.5),"snapshot","snapshot","W03",None),
    "spatial_G05_assumed_G1": (4,("gaussian",.5),("gaussian",1.),"snapshot","snapshot","W03",None),
    "spatial_G2_matched": (4,("gaussian",2.),("gaussian",2.),"snapshot","snapshot","W03",None),
    "spatial_G2_assumed_G1": (4,("gaussian",2.),("gaussian",1.),"snapshot","snapshot","W03",None),
    "spatial_C1_matched": (4,("compact",1.),("compact",1.),"snapshot","snapshot","W03",None),
    "spatial_C1_assumed_G1": (4,("compact",1.),("gaussian",1.),"snapshot","snapshot","W03",None),
    "covariance_mix_oracle": (4,("gaussian",1.),("gaussian",1.),"snapshot","snapshot","W_mix_oracle",None),
    "covariance_exp_population": (4,("gaussian",1.),("gaussian",1.),"snapshot","snapshot","W_exp_population",None),
    "covariance_exp_estimated": (4,("gaussian",1.),("gaussian",1.),"snapshot","snapshot","W_exp_estimated",None),
    "temporal_average_matched": (2,("gaussian",1.),("gaussian",1.),"average","average","W03","temporal_average"),
    "temporal_average_assumed_snapshot": (2,("gaussian",1.),("gaussian",1.),"average","snapshot","W03",None),
}


def test_exact_152_new_24_reused_paths_and_attempt_counts():
    design = build_design()
    assert design.counts == dict(new_paths=152,reused_paths=24,new_attempts=3800,
                                reused_attempts=600,contrasts=224)
    assert len(design.paths) == 176
    assert len({p.id for p in design.paths}) == 176
    assert len({c.id for c in design.contrasts}) == 224
    assert Counter(p.condition for p in design.paths) == {
        name:4*spec[0] for name,spec in EXPECTED.items()}
    expected_keys = {(condition,source,r,penalty) for condition,spec in EXPECTED.items()
                     for source in ('PG10','EC04') for r in range(1,spec[0]+1)
                     for penalty in ('L2','H1')}
    assert {(p.condition,p.source,p.replicate,p.penalty) for p in design.paths} == expected_keys
    assert all(p.alpha_exponents == tuple(-8+.5*j for j in range(25)) for p in design.paths)
    assert all(p.nodes == 73 and p.grid == p.truth_grid == 'G0' and p.tau_hours == .25
               for p in design.paths)
    assert tuple(FULL_TIMES_HOURS[j] for j in PRIMARY_TICKS) == PRIMARY_TIMES_HOURS


def test_exact_h_pairs_noise_and_no_accidental_factorial():
    design = build_design()
    for p in design.paths:
        repeats,true_kernel,assumed_kernel,true_time,assumed_time,weight,reuse_id = EXPECTED[p.condition]
        assert (p.true_h.spatial_kind,p.true_h.width_km) == true_kernel
        assert (p.inverse_h.spatial_kind,p.inverse_h.width_km) == assumed_kernel
        assert (p.true_h.temporal_kind,p.inverse_h.temporal_kind) == (true_time,assumed_time)
        for obs in (p.true_h,p.inverse_h):
            assert obs.window_hours == (1/3 if obs.temporal_kind == 'average' else None)
        assert p.weight == weight and p.replicate <= repeats
        assert (p.reuse.condition if p.reuse else None) == reuse_id
        assert p.noise.station_sd == (1.,1.5,2.,1.)
        assert p.stream.seed == 20260926
        if p.condition.startswith('covariance_'):
            assert p.stream.version == 3
            assert p.noise.family == 'station_exponential_mixture'
            assert p.true_h == p.inverse_h == ObservationSpec('gaussian',1.)
            lag = 1/3
            w = p.noise.fast_weight
            r1 = w*math.exp(-lag/p.noise.fast_hours)+(1-w)*math.exp(-lag/p.noise.slow_hours)
            assert r1 == pytest.approx(math.exp(-lag/design.population_length_hours),rel=1e-14)
        else:
            assert p.stream.version == 2 and p.noise.family == 'station_exponential'
            assert p.noise.lengths_hours == (.25,)*4
    # Нет варианта с истинным G1 и другим предполагаемым пространственным ядром, смешанного
    # семейства формы и ширины или изменения семейства ковариации в варианте с другим H.
    assert not any(p.true_h == ObservationSpec('gaussian',1.)
                   and p.inverse_h.spatial_kind == 'compact' for p in design.paths)


def test_reuse_references_are_supported_by_actual_frozen_configuration(project_root):
    # Проверяется только конфигурация; пути расчётов и контрольных записей не открываются.
    project = project_root
    config = json.loads((project/'experiments/source_recovery/configs/experiment.json').read_text('utf8'))
    conditions = {c['id']:c for c in config['conditions']}
    design = build_design()
    assert Counter(p.reuse.condition for p in design.reused_paths) == {'main':16,'temporal_average':8}
    for p in design.reused_paths:
        ref = p.reuse
        condition = conditions[ref.condition]
        assert ref.study == 'research_validation_v2'
        assert ref.source == p.source and p.source in condition['sources']
        assert ref.replicate == p.replicate and p.replicate in condition['replicates']
        assert ref.penalty == p.penalty and p.penalty in condition['penalties']
        assert condition['grid'] == p.grid and condition['truth_grid'] == p.truth_grid
        assert condition['nodes'] == p.nodes and condition['tau_hours'] == p.tau_hours
        assert condition['weight'] == p.weight and condition['noise'] == 'corr'
        assert p.noise.family == config['noise']['corr']['type']
        assert p.noise.station_sd == tuple(config['noise']['corr']['station_sd'])
        assert p.noise.lengths_hours == tuple(config['noise']['corr']['ell_hours'])
        assert condition['availability'] == 'full' and condition['mask'] == 'none'
        assert condition['relocation_km'] == 0
        assert condition['temporal_H'] == ('average20' if p.true_h.temporal_kind == 'average' else 'snapshot')
        assert p.true_h == p.inverse_h
        assert p.true_h.width_km == config['observations']['spatial_standard_deviation_km']
        assert asdict(p.stream) == config['stream']
        assert tuple(config['alpha']['exponents']) == p.alpha_exponents


def test_pairs_hold_data_and_other_factors_as_declared():
    design = build_design()
    paths = {p.id:p for p in design.paths}
    expected_counts = {'penalty':88,'assumed_spatial_H':48,'matched_spatial_design':48,
                       'assumed_temporal_H':8,'covariance_family':16,'covariance_estimation':16}
    assert Counter(c.factor for c in design.contrasts) == expected_counts
    assert len({c.key for c in design.contrasts}) == 21
    for c in design.contrasts:
        left,right = paths[c.left_path],paths[c.right_path]
        assert left.source == right.source and left.replicate == right.replicate
        assert left.standard_normal_key == right.standard_normal_key
        assert left.calibration_design_key == right.calibration_design_key
        if c.factor == 'matched_spatial_design':
            assert left.true_h == left.inverse_h and right.true_h == right.inverse_h
            assert left.true_h != right.true_h
            assert left.data_design_key != right.data_design_key  # Меняется порождающий оператор наблюдения.
        else:
            assert left.data_design_key == right.data_design_key
        if c.factor == 'penalty':
            assert (left.penalty,right.penalty) == ('L2','H1')
        else:
            assert left.penalty == right.penalty
    # Ключ генератора шума и калибровки не включает источник: шум общий для обоих источников.
    assert paths['spatial_G1_matched/PG10/r1/L2'].standard_normal_key == paths[
        'spatial_G2_matched/EC04/r1/H1'].standard_normal_key


def test_build_and_validation_are_deterministic_immutable_and_do_no_io(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('declarative design must not read files')
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'open',forbidden)
        patch.setattr(Path,'open',forbidden)
        first,second = build_design(),build_design()
        validate_design(first)
        assert first == second
    json.dumps(asdict(first),allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        first.paths = ()
    with pytest.raises(FrozenInstanceError):
        first.paths[0].stream.version = 900
    counts = first.counts
    counts['new_paths'] = -1
    assert first.counts['new_paths'] == 152
    sd,lengths = [1,1.5,2,1],[.25]*4
    noise = NoiseSpec('station_exponential',sd,lengths)
    sd[:],lengths[:] = [900]*4,[900]*4
    assert noise.station_sd == (1,1.5,2,1) and noise.lengths_hours == (.25,)*4


@pytest.mark.parametrize('change',['missing','duplicate','extra','wrong_h','wrong_stream','wrong_reuse'])
def test_changed_path_set_fails_closed(change):
    design = build_design()
    paths = list(design.paths)
    if change == 'missing':
        paths.pop()
    elif change == 'duplicate':
        paths.append(paths[0])
    elif change == 'extra':
        paths.append(replace(paths[0],condition='unregistered_extra',reuse=None))
    elif change == 'wrong_h':
        paths[0] = replace(paths[0],inverse_h=ObservationSpec('gaussian',2.))
    elif change == 'wrong_stream':
        paths[0] = replace(paths[0],stream=StreamSpec(20260926,3))
    else:
        paths[0] = replace(paths[0],reuse=None)
    with pytest.raises(ValueError):
        StudyDesign(paths,design.contrasts)


@pytest.mark.parametrize('change',['missing','duplicate','unknown_path','confounded','reversed','wrong_key'])
def test_changed_contrasts_cannot_hide_scope_or_multiple_factors(change):
    design = build_design()
    contrasts = list(design.contrasts)
    if change == 'missing':
        contrasts.pop()
    elif change == 'duplicate':
        contrasts.append(contrasts[0])
    elif change == 'unknown_path':
        contrasts[0] = replace(contrasts[0],right_path='missing')
    elif change == 'confounded':
        contrasts[0] = replace(contrasts[0],right_path='spatial_G2_matched/PG10/r1/H1')
    elif change == 'reversed':
        c = contrasts[0]
        contrasts[0] = replace(c,left_path=c.right_path,right_path=c.left_path)
    else:
        contrasts[0] = replace(contrasts[0],key='posthoc_key')
    with pytest.raises(ValueError):
        StudyDesign(design.paths,contrasts)


@pytest.mark.parametrize('width',[0,-1,float('nan'),float('inf'),True,'1',1+0j])
def test_observation_parameters_are_finite_and_numeric(width):
    with pytest.raises(ValueError):
        ObservationSpec('gaussian',width)


@pytest.mark.parametrize('kwargs',[dict(spatial_kind='other',width_km=1),
    dict(spatial_kind='gaussian',width_km=1,window_hours=1/3),
    dict(spatial_kind='gaussian',width_km=1,temporal_kind='average'),
    dict(spatial_kind='gaussian',width_km=1,temporal_kind='average',window_hours=float('inf'))])
def test_invalid_observation_combination(kwargs):
    with pytest.raises(ValueError):
        ObservationSpec(**kwargs)


@pytest.mark.parametrize('kwargs',[dict(replicate=0),dict(replicate=5),dict(replicate=True),
    dict(source='NEW-J2'),dict(penalty='TV'),dict(nodes=145),dict(grid='Gxt'),
    dict(tau_hours=float('nan')),dict(tau_hours=1.),dict(alpha_exponents=(-8.,0.,4.)),
    dict(alpha_exponents=(*ALPHA_EXPONENTS[:-1],float('inf'))),dict(condition='../escape')])
def test_invalid_path_parameters(kwargs):
    with pytest.raises(ValueError):
        replace(build_design().paths[0],**kwargs)


def test_noise_stream_and_reuse_validation():
    for seed,version in [(True,2),(20260926,0),(20260926,float('inf')),(20260926,'2')]:
        with pytest.raises(ValueError):
            StreamSpec(seed,version)
    for noise in [dict(family='station_exponential',station_sd=[1,2],lengths_hours=(.25,)*4),
                  dict(family='station_exponential',station_sd=[1,1,1,float('nan')],lengths_hours=(.25,)*4),
                  dict(family='station_exponential_mixture',station_sd=[1]*4,fast_hours=1,
                       slow_hours=1,fast_weight=.5),
                  dict(family='station_exponential_mixture',station_sd=[1]*4,fast_hours=.1,
                       slow_hours=1,fast_weight=1)]:
        with pytest.raises(ValueError):
            NoiseSpec(**noise)
    for condition,r in [('other',1),('temporal_average',3)]:
        with pytest.raises(ValueError):
            ReuseReference(condition,'PG10',r,'L2')
    p = build_design().paths[0]
    with pytest.raises(ValueError):
        replace(p,reuse=ReuseReference('main','EC04',1,'L2'))
    with pytest.raises(ValueError):
        ContrastSpec('x','unknown',p.id,p.id+'other')



def selected_configuration():
    return dict(version=4, selection=dict(path_ids=[
        "spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1",
        "spatial_G2_matched/PG10/r1/L2", "spatial_G2_matched/PG10/r1/H1"]),
        direct=dict(source_ids=["PG10"], domains=dict(D0="G0", D1="E06_D1_direct"),
            grids=dict(E06_D1_direct=dict(bounds_km=[-18., 12., -10., 10.], spacing_km=1., steps=168)),
            observation_ids=["G1_snapshot", "G05_snapshot", "G2_snapshot", "C1_snapshot", "G1_average20"],
            quadrature_spacings_km=[.25, .125, .0625]),
        analysis=dict(figures=[dict(id="matched", contrast_keys=["matched_spatial/G2"])]))


def test_resolver_preserves_full_catalogue_hash_and_counts():
    from adrkit.config.validation import digest
    from tests.experiments.observation_sensitivity.fixtures.paired_records import baseline_spec
    from experiments.observation_sensitivity.design import resolve_plan, design_from_record
    full = resolve_plan(dict(version=4), baseline_spec())
    assert digest(full["design"]) == "56ce94438a473d958d2363eac5ffe2a0c334cf24846a631dfc4182ba26a2778d"
    assert full["counts"] == dict(paths=176, contrasts=224, new_paths=152, reused_paths=24,
        new_fit_attempts=3800, reused_fit_attempts=600, inverse_groups=8, source_profiles=6)
    assert design_from_record(full["design"])["paths"] == build_design().paths


def test_selected_scope_keeps_scientific_descriptors_and_uses_baseline_g0():
    from tests.experiments.observation_sensitivity.fixtures.paired_records import baseline_spec
    from experiments.observation_sensitivity.design import resolve_plan
    spec = baseline_spec()
    spec["grids"]["G0"].update(spacing_km=1., steps=168)
    configuration = selected_configuration()
    plan = resolve_plan(configuration, spec)
    assert plan["counts"] == dict(paths=4, contrasts=4, new_paths=2, reused_paths=2,
        new_fit_attempts=50, reused_fit_attempts=50, inverse_groups=1, source_profiles=1)
    assert tuple(plan["expected_paths"]) == ("PG10/r1",)
    assert plan["baseline_groups"] == (("PG10", 1),)
    assert all(p in build_design().paths and len(p.alpha_exponents) == 25 and p.nodes == 73 for p in plan["paths"])
    assert plan["direct"]["grids"]["G0"] == spec["grids"]["G0"]
    assert plan["direct"]["grids"]["E06_D1_direct"]["spacing_km"] == 1.
    assert plan["direct"]["quadrature_spacings_km"] == [.25, .125, .0625]
    assert plan["paths"][0].reuse.condition == "main" and plan["paths"][0].weight == "W03"
    assert {c.key for c in plan["contrasts"]} == {"penalty/spatial_G1_matched", "penalty/spatial_G2_matched", "matched_spatial/G2"}


def test_absent_contrasts_and_explicit_empty_contrasts_are_distinct():
    from tests.experiments.observation_sensitivity.fixtures.paired_records import baseline_spec
    from experiments.observation_sensitivity.design import resolve_plan
    config = selected_configuration()
    assert len(resolve_plan(config, baseline_spec())["contrasts"]) == 4
    config["selection"]["contrast_ids"] = []
    config["analysis"]["figures"] = []
    plan = resolve_plan(config, baseline_spec())
    assert plan["contrasts"] == () and plan["figures"] == [] and plan["counts"]["contrasts"] == 0


@pytest.mark.parametrize("defect", ["unknown", "duplicate", "order", "missing_endpoint", "descriptor", "g0_override"])
def test_selected_plan_rejects_inconsistent_declarations(defect):
    from copy import deepcopy
    from tests.experiments.observation_sensitivity.fixtures.paired_records import baseline_spec
    from experiments.observation_sensitivity.design import resolve_plan, design_from_record
    spec = baseline_spec()
    config = selected_configuration()
    if defect == "unknown": config["selection"]["path_ids"][0] = "unknown"
    elif defect == "duplicate": config["selection"]["path_ids"].append(config["selection"]["path_ids"][0])
    elif defect == "order": config["selection"]["path_ids"].reverse()
    elif defect == "missing_endpoint":
        full = build_design()
        config["selection"]["contrast_ids"] = [full.contrasts[1].id]
    elif defect == "g0_override": config["direct"]["grids"]["G0"] = deepcopy(spec["grids"]["G0"])
    else:
        record = resolve_plan(config, spec)["design"]
        record["paths"][0]["inverse_h"]["width_km"] = 2.
        with pytest.raises(ValueError): design_from_record(record)
        return
    with pytest.raises(ValueError): resolve_plan(config, spec)


@pytest.mark.parametrize("value", [None, [], 1, "invalid"])
def test_plan_rejects_non_object_configuration_and_direct(value):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import baseline_spec
    from experiments.observation_sensitivity.design import resolve_plan
    with pytest.raises(ValueError, match="Configuration object"):
        resolve_plan(value, baseline_spec())
    config = selected_configuration()
    config["direct"] = value
    with pytest.raises(ValueError, match="direct plan must be an object"):
        resolve_plan(config, baseline_spec())


@pytest.mark.parametrize("value", [None, [], 1, "invalid"])
def test_plan_rejects_non_object_direct_grids(value):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import baseline_spec
    from experiments.observation_sensitivity.design import resolve_plan
    config = selected_configuration()
    config["direct"]["grids"] = value
    with pytest.raises(ValueError, match="Direct grids must be an object"):
        resolve_plan(config, baseline_spec())
