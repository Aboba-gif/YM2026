"""Проверки разделения выборок, фиксации выбора и продолжения расчёта."""
import adrkit.inverse.projected as inverse_projected
import experiments.source_comparison.calibration as comparison_calibration
import experiments.source_comparison.config as comparison_config
import experiments.source_recovery.checkpoints as recovery_checkpoints
import experiments.source_recovery.config as recovery_config
import experiments.source_recovery.driver as recovery_driver
import experiments.source_recovery.panels as recovery_panels
import experiments.source_recovery.run as recovery_run
from copy import deepcopy
from dataclasses import FrozenInstanceError
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from experiments.source_comparison.calibration import noise_covariance
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace
from adrkit.predictions import StateCertificate

from experiments.source_recovery.backend import disclose_history, make_observations, make_solver, Prediction, source_scores


def condition(**changes):
    value = dict(id="base",sources=["PG10"],replicates=[1],weight="W01",penalties=["L2","H1"],
        grid="G0",nodes=73,noise="corr",tau_hours=.25,truth_gamma=1/72,inverse_gamma=1/72,
        availability="full",mask="none",temporal_H="snapshot",relocation_km=0.)
    return dict(value,**changes)


def test_streams_are_independent_but_reproducible_and_noise_is_paired():
    spec = recovery_config.template_config(100)
    values = [recovery_panels.standard_panel(spec["stream"],1,p)[0] for p in recovery_config.PANEL_CODES]
    assert all(not np.array_equal(a,b) for i,a in enumerate(values) for b in values[i+1:])
    assert np.array_equal(values[0],recovery_panels.standard_panel(spec["stream"],1,"calibration")[0])
    assert not np.array_equal(values[0],recovery_panels.standard_panel(spec["stream"],2,"calibration")[0])
    covs = {name:noise_covariance(recovery_config.DENSE_TIMES,noise) for name,noise in spec["noise"].items()}
    assert np.array_equal(np.diag(covs["corr"]),np.diag(covs["iid_high"]))
    assert np.allclose(covs["iid_low"],.01*covs["iid_high"],atol=1e-17)
    high,ph = recovery_panels.draw_residual(spec["stream"],1,"fit",covs["iid_high"])
    low,pl = recovery_panels.draw_residual(spec["stream"],1,"fit",covs["iid_low"])
    assert ph["standard_normal_sha256"] == pl["standard_normal_sha256"]
    assert np.allclose(low,.1*high)


def test_masks_match_cardinality_and_covariance_is_restricted_before_inverse():
    spec=recovery_config.template_config(100)
    for kind in ("random2","block2"):
        rows,record=recovery_panels.selected_rows(condition(mask=kind),spec["stream"],3)
        assert len(rows)==28 and len(record["removed_primary_tick_indices"])==2
        assert np.array_equal(rows,recovery_panels.selected_rows(condition(mask=kind),spec["stream"],3)[0])
        assert all(np.array_equal(rows[:7],rows[s*7:(s+1)*7]-s*72) for s in range(4))
    covariance=np.array([[2.,.6,.3],[.6,1.,.5],[.3,.5,3.]])
    rows=np.array([0,2])
    metric=recovery_panels.restricted_metric(covariance,rows)
    expected=np.linalg.inv(covariance[np.ix_(rows,rows)])
    assert np.allclose(metric.apply_precision(np.ones(2)),expected@np.ones(2))
    assert not np.allclose(expected,np.linalg.inv(covariance)[np.ix_(rows,rows)])


def test_mass_normalized_spectrum_is_invariant_under_coefficient_reparameterization():
    rng=np.random.default_rng(311)
    white=rng.normal(size=(7,4))
    a=rng.normal(size=(4,4))
    gram=a.T@a+np.eye(4)
    transform=np.array([[2.,.2,0,0],[0,3.,.1,0],[0,0,.4,.3],[0,0,0,1.5]])
    first=recovery_panels.observation_diagnostics(white,gram)
    second=recovery_panels.observation_diagnostics(white@transform,transform.T@gram@transform)
    assert np.allclose(first["mass_normalized"]["singular_values"],second["mass_normalized"]["singular_values"],rtol=2e-13,atol=2e-14)
    assert not np.allclose(first["raw_coefficient_basis_dependent"]["singular_values"],second["raw_coefficient_basis_dependent"]["singular_values"])
    assert first["mass_normalized"]["protocol_relative_threshold"]==1e-10


def test_calibration_sees_only32_four_nine_and_dense_lift_preserves_marginal(monkeypatch):
    seen=[]
    original=comparison_calibration.fit_covariance
    def spy(residuals,times,**kwargs):
        seen.append((residuals.shape,np.array(times)))
        return original(residuals,times,**kwargs)
    monkeypatch.setattr(recovery_panels,"fit_covariance",spy)
    spec=recovery_config.template_config(100)
    covariance,record=recovery_panels.calibrate_dense(spec,1,"corr","W03")
    assert seen[0][0]==(32,4,9)
    assert np.array_equal(seen[0][1],recovery_config.DENSE_TIMES[recovery_config.PRIMARY_TICKS])
    assert covariance.shape==(288,288)
    assert record["calibration_shape"]==[32,4,9]
    p=record["parent_coarse_calibration"]
    expected=noise_covariance(recovery_config.DENSE_TIMES[recovery_config.PRIMARY_TICKS],dict(type="station_exponential",
        station_sd=np.sqrt(p["station_variances"]),ell_hours=p["ell_hours"]))
    assert np.allclose(covariance[np.ix_(recovery_config.PRIMARY_ROWS,recovery_config.PRIMARY_ROWS)],expected,rtol=2e-12,atol=2e-14)


def test_fixed_alpha_grid_and_tie_tolerances_are_explicit():
    spec=recovery_config.template_config(100)
    spec.update(frozen=True,conditions=[condition()])
    recovery_config.validate_config(spec)
    assert spec["alpha"]["policy"]=="fixed_common_grid"
    assert np.array_equal(recovery_config.BASE_EXPONENTS,np.arange(-8,4.01,.5))
    assert len(recovery_config.BASE_EXPONENTS)==25
    spec["alpha"]["extension_rounds"]=2
    with pytest.raises(ValueError,match="fixed25"):
        recovery_config.validate_config(spec)
    tied=[dict(alpha=a,exponent=i,accepted=True,selection_mse=1.+i*1e-12) for i,a in enumerate((1.,10.))]
    assert recovery_driver.select_candidate(tied)["alpha"]==10.
    assert recovery_driver.select_candidate(tied,atol=0,rtol=0)["alpha"]==1.


def test_checkpoint_rejects_changed_bindings_and_scoring_before_all_paths(tmp_path):
    p=tmp_path/"cp.json"
    cp=recovery_checkpoints.Checkpoint(p,{"code":"a"},["A","B"])
    with pytest.raises(ValueError,match="sealed"):
        cp.require_sealed()
    cp.record["paths"]["A"]={"finalized":True}
    with pytest.raises(ValueError,match="all paths"):
        cp.seal()
    with pytest.raises(ValueError,match="bindings"):
        recovery_checkpoints.Checkpoint(p,{"code":"b"},["A","B"])
    cp.record["paths"]["B"]={"finalized":True}
    cp.seal()
    cp.require_sealed()
    cp.record["paths"]["B"]["tampered"]=True
    with pytest.raises(ValueError,match="mismatch"):
        cp.require_sealed()


class LinearPrediction:
    """Линейный прогноз на 288 строках для проверок разделения выборок."""

    def __init__(self):
        self.basis=P1Basis(np.linspace(0,3,4),time_unit="h")
        self.domain=ArraySpace((self.basis.size,), axes=("coefficient",),
                               coordinates=(self.basis.knots,), units="dimensionless")
        self.codomain=ArraySpace((288,), axes=("observation",),
                                 coordinates=(np.arange(288),), units="ug/m^3")
        t=np.tile(recovery_config.DENSE_TIMES,4)
        self.matrix=np.column_stack([np.ones(288),t/3,np.sin(t)+1,np.cos(t)+1])
        self.trajectory=SimpleNamespace(max_scaled_residual=0.)
    def predict(self,p):
        return self.matrix@p
    def vjp(self,p,v):
        return self.matrix.T@v
    def jacobian(self,p):
        return self.matrix
    def invalidate(self):
        pass


class LinearBackend:
    """Линейный генератор и изменяемая проверочная ошибка для теста утечки."""

    def __init__(self,scored_truth=0.):
        self.scored_truth=scored_truth
    def truth(self,condition):
        return LinearPrediction().matrix@np.array([1.,.2,.3,.4]),0.
    def prediction(self,condition):
        return LinearPrediction()
    def score_source(self,condition,point):
        return dict(E_q=float(np.sum(np.asarray(point)**2))+self.scored_truth)


def _small_group(tmp_path,monkeypatch,*,alter_test=False,scored_truth=0.):
    spec=recovery_config.template_config(100)
    spec["conditions"]=[condition()]
    monkeypatch.setattr(recovery_driver,"calibrate_dense",lambda *args:(np.eye(288),{"fixed_test_metric":True}))
    cp=recovery_checkpoints.Checkpoint(tmp_path/"group.json",{"test":"immutable"},["base/L2","base/H1"])
    draw=recovery_panels.draw_residual
    touched=[]
    def guarded(*args,**kwargs):
        if args[2]=="test":
            cp.require_sealed()
            touched.append("test")
        values,record=draw(*args,**kwargs)
        if alter_test and args[2]=="test":
            values=values+500
        return values,record
    monkeypatch.setattr(recovery_driver,"draw_residual",guarded)
    recovery_driver.run_group(spec,"PG10",1,cp,LinearBackend(scored_truth))
    assert touched and cp.record["stage"]=="scored"
    return cp,spec


def test_whole_group_sealing_test_and_truth_canary_and_resume(tmp_path,monkeypatch):
    with monkeypatch.context() as m:
        first,spec=_small_group(tmp_path/"a",m)
    with monkeypatch.context() as m:
        second,_=_small_group(tmp_path/"b",m,alter_test=True,scored_truth=1000)
    for pid in first.record["paths"]:
        a,b=first.record["paths"][pid],second.record["paths"][pid]
        assert a["selected_exponent"]==b["selected_exponent"]
        for exponent in a["candidates"]:
            left,right=deepcopy(a["candidates"][exponent]),deepcopy(b["candidates"][exponent])
            assert left==right
        assert first.record["scores"][pid]["full_primary_test_rmse"]!=second.record["scores"][pid]["full_primary_test_rmse"]
    class ForbiddenBackend:
        def __getattr__(self,name):
            raise AssertionError("Completed resume must not regenerate or refit")
    assert recovery_driver.run_group(spec,"PG10",1,first,ForbiddenBackend())["stage"]=="scored"


@pytest.mark.parametrize("limited",[dict(availability="onlyKrAZ"),dict(mask="random2")])
def test_hidden_selection_rows_cannot_change_limited_candidate_grid_or_selection(tmp_path,monkeypatch,limited):
    spec=recovery_config.template_config(100)
    limited_condition=condition(id="limited",**limited)
    spec["conditions"]=[condition(),limited_condition]
    retained,_=recovery_panels.selected_rows(limited_condition,spec["stream"],1)
    hidden=np.setdiff1d(recovery_config.PRIMARY_ROWS,retained)
    assert len(hidden)>0
    monkeypatch.setattr(recovery_driver,"calibrate_dense",lambda *args:(np.eye(288),{"fixed_test_metric":True}))
    class HiddenRowsBackend(LinearBackend):
        def __init__(self,shift):
            super().__init__()
            self.shift=shift
        def truth(self,c):
            values,residual=super().truth(c)
            values[hidden]+=self.shift
            return values,residual
    expected=[c["id"]+"/"+arm for c in spec["conditions"] for arm in c["penalties"]]
    checkpoints=[]
    for index,shift in enumerate((0.,500.)):
        cp=recovery_checkpoints.Checkpoint(tmp_path/f"hidden_{index}.json",{"same":"binding"},expected)
        recovery_driver.run_group(spec,"PG10",1,cp,HiddenRowsBackend(shift))
        checkpoints.append(cp)
    first,second=[cp.record for cp in checkpoints]
    assert first["exponents"]==second["exponents"]==list(recovery_config.BASE_EXPONENTS)
    for arm in ("L2","H1"):
        a,b=first["paths"]["limited/"+arm],second["paths"]["limited/"+arm]
        assert a["selected_exponent"]==b["selected_exponent"]
        assert a["alpha_reference"]==b["alpha_reference"]
        assert len(a["candidates"])==len(b["candidates"])==25
        for exponent in a["candidates"]:
            left,right=deepcopy(a["candidates"][exponent]),deepcopy(b["candidates"][exponent])
            assert left==right
        assert a["provenance"]["selection_y_sha256"]==b["provenance"]["selection_y_sha256"]
        assert first["paths"]["base/"+arm]["provenance"]["selection_y_sha256"]!=second["paths"]["base/"+arm]["provenance"]["selection_y_sha256"]


def test_history_has_no_future_access_and_temporal_average_is_actual_integral():
    class HistorySpy:
        def value(self,t):
            assert t<0
            return 2.
        def integral(self,a,b):
            assert b<=0
            return 2*(b-a)
    history=disclose_history(HistorySpy(),np.linspace(0,3.5,85),-.5)
    assert all(b<=0 for a,b in history.intervals)
    with pytest.raises(ValueError,match="undisclosed"):
        history.integral(0,.1)
    assert not hasattr(history,"source")
    spec=recovery_config.template_config(100)
    spec["grids"]["tiny"]=dict(bounds_km=[-1,1,-1,1],spacing_km=.5,steps=84)
    solver=make_solver(spec,"tiny",0.)
    _,weights=make_observations(spec,solver,condition(temporal_H="average20"))
    physical=solver.times-.5
    assert np.allclose(weights@np.ones(len(physical)),1.)
    assert np.allclose((weights@physical)[recovery_config.PRIMARY_TICKS],recovery_config.DENSE_TIMES[recovery_config.PRIMARY_TICKS]-1/6)


def test_template_requires_explicit_freeze_and_conditions():
    spec=recovery_config.template_config(100)
    with pytest.raises(ValueError,match="frozen"):
        recovery_config.validate_config(spec)
    spec["frozen"]=True
    with pytest.raises(ValueError,match="conditions"):
        recovery_config.validate_config(spec)
    spec["conditions"]=[condition()]
    recovery_config.validate_config(spec)


@pytest.mark.parametrize("resources", [None, dict(workers=4, max_trajectory_bytes=1)])
def test_scientific_configuration_validation_does_not_impose_execution_limits(resources):
    spec = recovery_config.template_config(100)
    assert "resources" not in spec
    spec.update(frozen=True, conditions=[condition()])
    if resources is not None:
        spec["resources"] = resources
    recovery_config.validate_config(spec)


@pytest.mark.parametrize("resources", [None, dict(workers=4, max_trajectory_bytes=1)])
def test_solver_construction_uses_science_without_a_trajectory_budget(resources):
    spec = recovery_config.template_config(100)
    spec["grids"]["tiny"] = dict(bounds_km=[-1, 1, -1, 1], spacing_km=.5, steps=84)
    if resources is not None:
        spec["resources"] = resources
    solver = make_solver(spec, "tiny", 0.)
    assert np.array_equal(solver.times, np.linspace(0., 3.5, 85))


def test_interrupted_resume_keeps_finished_candidates_and_failed_fits_are_not_replaced(tmp_path,monkeypatch):
    spec=recovery_config.template_config(100)
    spec["conditions"]=[condition()]
    monkeypatch.setattr(recovery_driver,"calibrate_dense",lambda *args:(np.eye(288),{"fixed_test_metric":True}))
    cp=recovery_checkpoints.Checkpoint(tmp_path/"resume.json",{"binding":"same"},["base/L2","base/H1"])
    calls=[]
    def interrupt(*args,**kwargs):
        calls.append(args[4])
        if len(calls)==4:
            raise KeyboardInterrupt("simulated process interruption")
        if len(calls)==2:
            raise RuntimeError("simulated terminal numerical failure")
        return inverse_projected.fit(*args,**kwargs)
    with pytest.raises(KeyboardInterrupt):
        recovery_driver.run_group(spec,"PG10",1,cp,LinearBackend(),fitter=interrupt)
    completed=deepcopy(cp.record["paths"]["base/L2"]["candidates"])
    assert len(completed)==3
    assert completed["-7.5"]["status"]=="numerical_failure"
    resumed=recovery_checkpoints.Checkpoint(cp.path,{"binding":"same"},["base/L2","base/H1"])
    seen=[]
    def record(*args,**kwargs):
        seen.append((comparison_calibration.array_hash(args[3]),args[4]))
        return inverse_projected.fit(*args,**kwargs)
    recovery_driver.run_group(spec,"PG10",1,resumed,LinearBackend(),fitter=record)
    current=resumed.record["paths"]["base/L2"]
    for key,value in completed.items():
        assert current["candidates"][key]==value
    assert not current["complete_path"] and not current["procedure_accepted"]
    assert len(seen)==sum(r["candidate_count"] for r in resumed.record["paths"].values())-3


def test_single_fits_reuse_physical_alpha_and_only_explicit_check_gets_zero_start(tmp_path,monkeypatch):
    spec=recovery_config.template_config(100)
    spec["conditions"]=[condition()]
    spec["single_fits"]=[dict(id=kind,condition="base",baseline_condition="base",kind=kind,
        penalty="L2",sources=["PG10"],replicates=[1]) for kind in ("fixed_alpha","zero_start")]
    monkeypatch.setattr(recovery_driver,"calibrate_dense",lambda *args:(np.eye(288),{"fixed_test_metric":True}))
    expected=["base/L2","base/H1","single/fixed_alpha","single/zero_start"]
    cp=recovery_checkpoints.Checkpoint(tmp_path/"single.json",{"immutable":True},expected)
    zero_calls=[]
    def checked(*args,**kwargs):
        if "initial_point" in kwargs:
            assert np.array_equal(kwargs["initial_point"],np.zeros(4))
            zero_calls.append(args[4])
        return inverse_projected.fit(*args,**kwargs)
    recovery_driver.run_group(spec,"PG10",1,cp,LinearBackend(),fitter=checked)
    base=cp.record["paths"]["base/L2"]
    alpha=base["candidates"][str(base["selected_exponent"])]["alpha"]
    assert zero_calls==[alpha]
    assert cp.record["paths"]["single/fixed_alpha"]["alpha"]==alpha
    assert cp.record["paths"]["single/zero_start"]["alpha"]==alpha
    assert all(r["finalized"] for r in cp.record["paths"].values())


def test_calibration_failure_remains_explicit_and_does_not_resample(tmp_path,monkeypatch):
    spec=recovery_config.template_config(100)
    spec["conditions"]=[condition()]
    calls=[]
    def failure(*args):
        calls.append(args[1:])
        raise comparison_calibration.CalibrationFailure("unidentifiable endpoint")
    monkeypatch.setattr(recovery_driver,"calibrate_dense",failure)
    cp=recovery_checkpoints.Checkpoint(tmp_path/"calfail.json",{"immutable":True},["base/L2","base/H1"])
    recovery_driver.run_group(spec,"PG10",1,cp,LinearBackend())
    assert len(calls)==1 and cp.record["scores"]=={}
    assert all(r["calibration_failure"] and r["candidate_count"]==0 and
               not r["procedure_accepted"] for r in cp.record["paths"].values())


@pytest.mark.parametrize("temporal",["snapshot","average20"])
def test_prediction_adapter_has_correct_adjoint_and_no_source_retention(temporal):
    class Known:
        def value(self,t):
            assert t<0
            return 1.
        def integral(self,a,b):
            assert b<=0
            return b-a
    spec=recovery_config.template_config(100)
    spec["grids"]["tiny"]=dict(bounds_km=[-1,1,-1,1],spacing_km=.5,steps=84)
    spec["observations"]["centers_km"]=[[0,0],[.2,0],[0,.2],[.2,.2]]
    c=condition(grid="tiny",nodes=4,temporal_H=temporal)
    solver=make_solver(spec,"tiny",1/72)
    source=Known()
    reference=weakref.ref(source)
    history=disclose_history(source,solver.times,-.5)
    p=Prediction(spec,c,solver,history)
    assert not any(hasattr(p,name) for name in ("source","load","solver"))
    assert p.trajectory is None
    del source
    gc.collect()
    assert reference() is None
    at=np.array([.1,.2,.3,.2]);direction=np.array([.2,-.3,.1,.4])
    dual=np.sin(np.arange(288))
    jacobian=p.jacobian(at)
    gradient=p.vjp(at,dual)
    certificate=p.trajectory
    assert isinstance(certificate,StateCertificate) and not hasattr(certificate,"states")
    with pytest.raises(FrozenInstanceError):
        certificate.max_scaled_residual=999.
    assert np.allclose(gradient,jacobian.T@dual,atol=2e-8,rtol=1e-8)
    eps=1e-5
    difference=(p.predict(at+eps*direction)-p.predict(at-eps*direction))/(2*eps)
    assert np.allclose(difference,jacobian@direction,rtol=2e-6,atol=2e-6)
    p.invalidate()
    assert p.trajectory is None


@pytest.mark.parametrize("shape",["jump","cosine"])
def test_scored_source_error_matches_independent_piecewise_quadrature(shape):
    from scipy.integrate import quad
    from experiments.source_recovery.sources import TwoPulse
    source=TwoPulse(shape,mass=73.)
    basis=P1Basis(np.linspace(0,3,8),time_unit="h")
    coefficients=np.linspace(.1,.3,8)
    record=source_scores(source,basis,coefficients,100.)
    edges=np.unique(np.r_[basis.knots,source.events])
    squared=sum(quad(lambda t:(100*np.interp(t,basis.knots,coefficients)-source.value(t))**2,
                     a,b,epsabs=1e-9,epsrel=1e-11)[0] for a,b in zip(edges[:-1],edges[1:]))
    assert record["E_q"]==pytest.approx(np.sqrt(squared)/(100*np.sqrt(3)),rel=1e-11)


@pytest.mark.parametrize("order",[("G0","Gxt"),("Gxt","G0")])
def test_truth_grid_override_is_validated_and_has_independent_cache_entry(monkeypatch,order):
    from experiments.source_recovery import backend as backend
    from experiments.source_recovery.sources import TwoPulse
    spec=recovery_config.template_config(100)
    assert spec["truth_grid"]=="G0"
    spec["frozen"]=True
    spec["conditions"]=[condition(truth_grid="missing")]
    with pytest.raises(ValueError,match="truth grid"):
        recovery_config.validate_config(spec)
    spec["grids"]["G0"]=dict(bounds_km=[-1,1,-1,1],spacing_km=.5,steps=84)
    spec["grids"]["Gxt"]=dict(bounds_km=[-1,1,-1,1],spacing_km=.5,steps=168)
    observed=[]
    original=backend.make_solver
    def spy(s,grid,gamma):
        observed.append(grid)
        return original(s,grid,gamma)
    monkeypatch.setattr(backend,"make_solver",spy)
    model=backend.ProductionBackend(spec,TwoPulse("jump"))
    values={name:model.truth(condition(truth_grid=name)) for name in order}
    assert model.truth(condition()) is values["G0"]
    assert model.truth(condition(truth_grid="Gxt")) is values["Gxt"]
    assert observed==list(order) and len(model.truth_cache)==2


SOURCE_HASHES = {
    "PG10":"01a6f9dbdc276ee616f94f95c7f158087cb0ed08799dd374053b9143c5bfee80",
    "SB150":"54f88dcdbfb8d3dbbc6c1db0fdd47ef4a1fe12a68971c62190431e93b259e2ba",
    "EC04":"c089deb36b9882cb8ba4d1b48926138e17b3a345d8d97f70b86ecadf889766ee",
    "EC06":"de058e0b94a106e0689790626a62d9ad8a109ab1e97d90d9ff4fb8729eb008fb",
    "NEW-J2":"57d91e9fbc86bdf12832e469eae912c00489ceec2e4f8700ab0db40cab5b3a47",
    "NEW-S2":"8ff8d0131b349647f7cf18b2af22873bc244d05efe84563a00d7590d3e643d2f",
}


@pytest.fixture(scope="module")
def real_scaled_sources(project_root):
    from experiments.source_recovery.sources import make_sources
    project=project_root
    binding=comparison_config.load_protocol(project/"experiments/source_comparison/configs/protocol.json",
        expected_sha256=recovery_config.template_config(100)["source_protocol"]["sha256"])
    return make_sources(binding,mass=172.0876693725586)


@pytest.mark.parametrize("name",list(SOURCE_HASHES))
def test_real_source_metadata_hash_uses_json_boundary_without_loosening_validator(name,real_scaled_sources):
    from adrkit.errors import ConfigError
    from experiments.source_recovery.sources import source_record
    source=real_scaled_sources[name]
    raw=source_record(source)
    normalized=json.loads(json.dumps(raw,allow_nan=False))
    before_values=source.value(np.array([-.5,0.,.4,1.5,3.])).copy()
    assert recovery_run.source_record_hash(source)==SOURCE_HASHES[name]==recovery_checkpoints.canonical_hash(normalized)
    assert np.array_equal(before_values,source.value(np.array([-.5,0.,.4,1.5,3.])))
    if name in ("EC04","EC06"):
        assert isinstance(raw["definition"]["amplitudes"],tuple)
        assert isinstance(raw["definition"]["decay_hours"],tuple)
        assert type(raw["unknown_mass"]) is np.float64
        with pytest.raises(ConfigError,match="JSON"):
            recovery_checkpoints.canonical_hash(raw)
    else:
        assert recovery_checkpoints.canonical_hash(raw)==recovery_run.source_record_hash(source)


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),object()])
def test_source_metadata_boundary_does_not_stringify_invalid_values(monkeypatch,bad):
    from experiments.source_recovery import sources
    monkeypatch.setattr(sources,"source_record",lambda source:{"invalid":bad})
    with pytest.raises((TypeError,ValueError)):
        recovery_run.source_record_hash(object())


@pytest.mark.parametrize("requested",[None,*SOURCE_HASHES])
def test_run_initialization_checks_all_source_hashes_before_any_numerical_work(tmp_path,monkeypatch,requested, project_root):
    from experiments.source_recovery import backend as experiment_backend
    project=project_root
    spec=recovery_config.template_config(172.0876693725586)
    spec.update(frozen=True,replicates=[1],conditions=[condition(sources=list(SOURCE_HASHES))])
    spec["source_protocol"]["path"]=str(project/"experiments/source_comparison/configs/protocol.json")
    output=tmp_path/"metadata-bootstrap-output"
    assert not output.exists()
    spec["output"]=str(output)
    path=tmp_path/"bootstrap.json"
    path.write_text(json.dumps(spec),encoding="utf-8")
    hashes=[]
    original=recovery_run.source_record_hash
    def observed_hash(source):
        value=original(source)
        hashes.append(value)
        return value
    events=[]
    def manifest_only(destination,record):
        assert hashes==list(SOURCE_HASHES.values())
        assert destination==output/"run.json"
        events.append("manifest")
    class MemoryCheckpoint:
        def __init__(self,path,bindings,expected_paths):
            assert len(hashes)==6
            assert bindings["source_record_sha256"]==SOURCE_HASHES[bindings["source"]]
            events.append("checkpoint:"+bindings["source"])
    class NoNumericsBackend:
        def __init__(self,*args):
            assert len(hashes)==6
    def no_group(spec,name,replicate,*args):
        assert len(hashes)==6 and replicate==1
        events.append("group:"+name)
    monkeypatch.setattr(recovery_run,"source_record_hash",observed_hash)
    monkeypatch.setattr(recovery_run,"atomic_json",manifest_only)
    monkeypatch.setattr(recovery_run,"Checkpoint",MemoryCheckpoint)
    monkeypatch.setattr(recovery_run,"run_group",no_group)
    monkeypatch.setattr(experiment_backend,"ProductionBackend",NoNumericsBackend)
    assert recovery_run.run_experiment(path,source_id=requested)==output
    names=list(SOURCE_HASHES) if requested is None else [requested]
    assert events==["manifest"]+[event for name in names for event in ("checkpoint:"+name,"group:"+name)]
    assert not list(output.rglob("*.json"))
    locks = [output/".run.lock", *(output/name/"replicate_1.json.lock" for name in names)]
    assert {p for p in output.rglob("*") if p.is_file()} == set(locks)
    from experiments.file_locks import FileLock
    for lock in locks:
        with FileLock(lock):
            pass


@pytest.mark.parametrize("residual,local_accepted,accepted", [
    (0., True, True), (1e-11, True, True), (1.1e-11, True, False),
    (-1e-12, True, False), (0., False, False),
])
def test_recovery_candidate_uses_real_state_residual_and_serializes_arrays(
        tmp_path, monkeypatch, residual, local_accepted, accepted):
    spec = recovery_config.template_config(100)
    spec["conditions"] = [condition()]
    monkeypatch.setattr(recovery_driver, "calibrate_dense",
                        lambda *args: (np.eye(288), {"fixed_test_metric": True}))
    class ResidualBackend(LinearBackend):
        def prediction(self, condition):
            prediction = LinearPrediction()
            prediction.trajectory = SimpleNamespace(max_scaled_residual=residual)
            return prediction
    cp = recovery_checkpoints.Checkpoint(tmp_path/"residual.json", {"test": "residual"},
                                        ["base/L2", "base/H1"])
    calls = []
    def estimate(*args, **options):
        calls.append(args[4])
        if len(calls) == 2:
            raise KeyboardInterrupt("after one committed candidate")
        result = inverse_projected.fit(*args, **options)
        assert isinstance(result["coefficients"], np.ndarray)
        assert isinstance(result["prediction"], np.ndarray)
        result["accepted"] = local_accepted
        return result
    with pytest.raises(KeyboardInterrupt, match="one committed"):
        recovery_driver.run_group(spec, "PG10", 1, cp, ResidualBackend(), fitter=estimate)
    candidate = cp.record["paths"]["base/L2"]["candidates"]["-8.0"]
    assert candidate["accepted"] is accepted
    assert candidate["forward_residual"] == residual
    assert isinstance(candidate["coefficients"], list)
    assert isinstance(candidate["prediction"], list)
    assert not {"forward_calls", "adjoint_calls", "seconds"} & candidate.keys()
    saved = json.loads(cp.path.read_text(encoding="utf-8"))
    assert saved["paths"]["base/L2"]["candidates"]["-8.0"] == candidate


def test_atomic_json_owns_distinct_temporaries_without_touching_existing_tmp(tmp_path, monkeypatch):
    sentinel = tmp_path / "unrelated.json"
    sentinel.write_bytes(b"unrelated file stays unchanged")
    target = tmp_path / "checkpoint.json"
    legacy_tmp = target.with_name(target.name + ".tmp")
    os.link(sentinel, legacy_tmp)
    replacements, synced = [], []
    real_replace, real_sync = os.replace, os.fsync
    def sync(fd):
        assert not os.get_inheritable(fd)
        synced.append(fd)
        return real_sync(fd)
    def replace(source, destination):
        source = Path(source)
        assert source.parent == target.parent and source != legacy_tmp
        assert Path(destination) == target and len(synced) == len(replacements) + 1
        assert json.loads(source.read_text(encoding="utf-8")) == {"value": len(replacements), "name": "Источник"}
        replacements.append(source)
        return real_replace(source, destination)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    for value in (0, 1):
        record = {"value": value, "name": "Источник"}
        recovery_checkpoints.atomic_json(target, record)
        expected = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        assert target.read_bytes() == expected
    assert len(set(replacements)) == 2 and len(synced) == 2
    assert not any(path.exists() for path in replacements)
    assert legacy_tmp.read_bytes() == sentinel.read_bytes() == b"unrelated file stays unchanged"


@pytest.mark.parametrize("failure", ["serialization", "fsync", "replace"])
def test_atomic_json_failure_preserves_committed_bytes_and_closes_own_temp(tmp_path, monkeypatch, failure):
    target = tmp_path / "checkpoint.json"
    target.write_bytes(b'{"committed": true}\n')
    original = target.read_bytes()
    opened = []
    real_mkstemp = recovery_checkpoints.tempfile.mkstemp
    def temporary(**options):
        fd, name = real_mkstemp(**options)
        opened.append((fd, Path(name)))
        return fd, name
    def failed(*args):
        raise OSError("Injected persistence failure")
    monkeypatch.setattr(recovery_checkpoints.tempfile, "mkstemp", temporary)
    record = {"value": object()} if failure == "serialization" else {"value": 3}
    if failure == "fsync":
        monkeypatch.setattr(os, "fsync", failed)
    elif failure == "replace":
        monkeypatch.setattr(os, "replace", failed)
    with pytest.raises((TypeError, OSError)):
        recovery_checkpoints.atomic_json(target, record)
    assert target.read_bytes() == original and len(opened) == 1
    assert not opened[0][1].exists()
    with pytest.raises(OSError):
        os.fstat(opened[0][0])


@pytest.mark.parametrize("input_case", [None, "config", "protocol", "duplicate_protocol"])
def test_run_manifest_binds_the_config_and_protocol_bytes_used(
        tmp_path, monkeypatch, project_root, input_case):
    from hashlib import sha256
    from experiments.source_recovery import backend as experiment_backend

    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = (project_root / "experiments/source_comparison/configs/protocol.json").read_bytes()
    protocol_path.write_bytes(protocol_bytes)
    spec = recovery_config.template_config(100.)
    output = tmp_path / "results"
    spec.update(frozen=True, sources=["PG10"], replicates=[1],
                conditions=[condition()], output=str(output))
    spec["source_protocol"]["path"] = str(protocol_path)
    if input_case == "duplicate_protocol":
        spec["input_files"] = [dict(path=str(protocol_path), sha256=sha256(protocol_bytes).hexdigest())]
    config_path = tmp_path / "experiment.json"
    config_bytes = (json.dumps(spec, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    config_path.write_bytes(config_bytes)
    original_read = Path.read_bytes
    original_validate = recovery_run.validate_config
    original_load = recovery_run.load_protocol
    reads = {config_path: 0, protocol_path: 0}
    used_configs, groups = [], []

    def read_bytes(path):
        if path in reads:
            reads[path] += 1
        return original_read(path)

    def validate(configuration):
        original_validate(configuration)
        if input_case == "config":
            replacement = dict(spec, source_mass=200.)
            config_path.write_text(json.dumps(replacement), encoding="utf-8")

    def load(path, **options):
        binding = original_load(path, **options)
        if input_case == "protocol":
            protocol_path.write_bytes(protocol_bytes + b"\n")
        return binding

    class NoNumericsBackend:
        def __init__(self, configuration, source):
            used_configs.append(configuration)
            assert source.integral(0., 3.) == pytest.approx(spec["source_mass"])

    def no_group(configuration, source, replicate, checkpoint, backend):
        groups.append((configuration, source, replicate, checkpoint.record["expected_paths"]))
        return checkpoint.record

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(recovery_run, "validate_config", validate)
    monkeypatch.setattr(recovery_run, "load_protocol", load)
    monkeypatch.setattr(experiment_backend, "ProductionBackend", NoNumericsBackend)
    monkeypatch.setattr(recovery_run, "run_group", no_group)
    assert recovery_run.run_experiment(config_path) == output
    assert reads == {config_path: 1, protocol_path: 1}
    raw_manifest = original_read(output / "run.json")
    manifest = json.loads(raw_manifest.decode("utf-8"))
    assert manifest["configuration"] == spec
    assert manifest["bindings"]["config_sha256"] == sha256(config_bytes).hexdigest()
    assert manifest["bindings"]["input_files"] == {str(protocol_path): sha256(protocol_bytes).hexdigest()}
    assert used_configs == [spec]
    assert groups == [(spec, "PG10", 1, ["base/L2", "base/H1"])]
    expected = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    assert raw_manifest == expected
    checkpoint = json.loads(original_read(output / "PG10/replicate_1.json"))
    for key, value in manifest["bindings"].items():
        assert checkpoint["bindings"][key] == value
    assert checkpoint["expected_paths"] == ["base/L2", "base/H1"]
    assert checkpoint["exponents"] == list(recovery_config.BASE_EXPONENTS)
    assert checkpoint["paths"] == {} and checkpoint["stage"] == "estimating"
    assert "scores" not in checkpoint
    for path, initial in ((config_path, config_bytes), (protocol_path, protocol_bytes)):
        if input_case == ("config" if path == config_path else "protocol"):
            assert original_read(path) != initial
            assert sha256(original_read(path)).hexdigest() != sha256(initial).hexdigest()
        else:
            assert original_read(path) == initial


@pytest.mark.parametrize("duplicate_input", ["config", "protocol"])
def test_duplicate_input_cannot_relabel_a_loaded_buffer_with_later_bytes(
        tmp_path, monkeypatch, project_root, duplicate_input):
    from hashlib import sha256
    from experiments.source_recovery import backend as experiment_backend

    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = (project_root / "experiments/source_comparison/configs/protocol.json").read_bytes()
    protocol_path.write_bytes(protocol_bytes)
    config_path = tmp_path / "experiment.json"
    target = config_path if duplicate_input == "config" else protocol_path
    later_bytes = b"a different configuration buffer\n" if duplicate_input == "config" else protocol_bytes + b"\n"
    spec = recovery_config.template_config(100.)
    output = tmp_path / "results"
    spec.update(frozen=True, sources=["PG10"], replicates=[1],
                conditions=[condition()], output=str(output),
                input_files=[dict(path=str(target), sha256=sha256(later_bytes).hexdigest())])
    spec["source_protocol"]["path"] = str(protocol_path)
    config_bytes = json.dumps(spec).encode("utf-8")
    config_path.write_bytes(config_bytes)
    original_read = Path.read_bytes
    original_validate = recovery_run.validate_config
    original_load = recovery_run.load_protocol
    reads = {config_path: 0, protocol_path: 0}

    def read_bytes(path):
        if path in reads:
            reads[path] += 1
        return original_read(path)

    def validate(configuration):
        original_validate(configuration)
        if duplicate_input == "config":
            config_path.write_bytes(later_bytes)

    def load(path, **options):
        binding = original_load(path, **options)
        if duplicate_input == "protocol":
            protocol_path.write_bytes(later_bytes)
        return binding

    def no_numerics(*args, **options):
        raise AssertionError("Input binding must reject before sources, backend or group")

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(recovery_run, "validate_config", validate)
    monkeypatch.setattr(recovery_run, "load_protocol", load)
    monkeypatch.setattr(recovery_run, "source_record_hash", no_numerics)
    monkeypatch.setattr(experiment_backend, "ProductionBackend", no_numerics)
    monkeypatch.setattr(recovery_run, "run_group", no_numerics)
    with pytest.raises(ValueError, match="Input hash changed") as error:
        recovery_run.run_experiment(config_path)
    assert str(error.value) == f"Input hash changed: {target.name}"
    assert reads == {config_path: 1, protocol_path: 1}
    assert original_read(target) == later_bytes
    initial = config_bytes if duplicate_input == "config" else protocol_bytes
    assert sha256(initial).hexdigest() != sha256(later_bytes).hexdigest()
    assert not output.exists()
