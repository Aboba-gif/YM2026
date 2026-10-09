"""Синтетические прямые записи с искусственными сигналами и матрицами наблюдения."""
import json

import numpy as np
import pytest

from adrkit.config.validation import digest
from experiments.source_comparison.truth import FiniteRelease
from experiments.source_recovery.config import template_config
from experiments.source_recovery.sources import ScaledSource, source_record
from experiments.source_recovery.run import source_record_hash
from experiments.observation_sensitivity.admission import scientific_configuration
from experiments.observation_sensitivity import direct
from experiments.observation_sensitivity.design import build_design, FULL_TIMES_HOURS
from experiments.observation_sensitivity.kernels import GaussianKernel, CompactKernel, spatial_weights
from experiments.observation_sensitivity.operators import temporal_weights
from tests.experiments.observation_sensitivity.fixtures.paired_records import make_admission, seal


class FakeFields:
    """Искусственные сигналы для прямых записей, включая log_width_derivative=True."""
    weights = {}
    def __init__(self, spec, source, *, source_id):
        self.spec, self.source_id = spec, source_id
        self.primed = set()

    def observation_weights(self, obs, *, grid_name):
        key = grid_name, obs
        if key not in self.weights:
            grid = self.spec['grids'][grid_name]
            kernel = (CompactKernel if obs.spatial_kind == 'compact' else GaussianKernel)(obs.width_km)
            space = spatial_weights(kernel, direct.interior_nodes(grid['bounds_km'], .25),
                                    self.spec['observations']['centers_km'], .25**2)
            time = temporal_weights(np.linspace(0., 3.5, 337), FULL_TIMES_HOURS,
                origin_hours=-.5, kind=obs.temporal_kind, window_hours=obs.window_hours)
            self.weights[key] = space, time
        return self.weights[key]

    def prime_truth(self, **kwargs):
        grid = kwargs['grid_name']
        assert grid not in self.primed
        self.primed.add(grid)

    def project_truth(self, obs, *, grid_name, log_width_derivative=False):
        assert grid_name in self.primed
        shift = .01 if grid_name == direct.D1_GRID else 0.
        signal = np.linspace(0., 1., 288) + obs.width_km + shift
        if log_width_derivative:
            signal *= 2.
        return signal, 1e-12

@pytest.fixture(scope='module')
def direct_template():
    spec = template_config(100.)
    sources = {name: ScaledSource(FiniteRelease(.5, 1., 100.), 1.) for name in direct.SOURCE_IDS}
    source_records = {name: dict(record=json.loads(json.dumps(source_record(source))),
                                sha256=source_record_hash(source)) for name, source in sources.items()}
    admission = make_admission(spec, source_records, build_design())
    science = scientific_configuration(admission['baseline']['configuration'])
    # Эталонная квадратура аналитического ядра вычисляется один раз; значения полей искусственные.
    result = direct.collect_direct(science, sources, backend_factory=FakeFields)
    freeze = seal(dict(schema='ym2026.observation_sensitivity.freeze', version=3,
        admission=admission, admission_sha256=digest(admission), frozen_at='2026-09-27T00:00:00+00:00'))
    journal = seal(dict(schema='ym2026.observation_sensitivity.direct_journal', version=3, status='completed',
        admission_sha256=digest(admission), freeze_sha256=freeze['content_sha256'],
        run_id='12345678-1234-1234-1234-123456789012',
        started_at='2026-09-27T00:01:00+00:00', finished_at='2026-09-27T00:02:00+00:00', result=result))
    return freeze, journal
