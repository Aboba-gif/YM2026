"""Аналитические профили: интегралы, предыстория и параметры."""
import json
import numpy as np
import pytest
from scipy.integrate import quad
from experiments.source_comparison.config import load_protocol
from experiments.source_comparison.truth import build_truth
from experiments.source_recovery.sources import TwoPulse, make_sources, first_moment, unknown_L2_squared


@pytest.fixture(scope="module")
def binding(project_root):
    path = project_root / "experiments/source_comparison/configs/protocol.json"
    config = json.loads((path.parent / "experiment.json").read_bytes())
    return load_protocol(path, expected_sha256=config["protocol_sha256"])


@pytest.mark.parametrize('name', ['PG10','SB150','EC04','EC06','NEW-J2','NEW-S2'])

def test_analytic_integrals_and_moments_against_adaptive_quadrature(binding, name):

    source = make_sources(binding, mass=271.4)[name]

    assert source.integral(0.,3.) == pytest.approx(271.4, rel=2e-14)

    cuts = sorted(set([-.5, 0., 3., *source.events]))

    for left, right in zip(cuts[:-1], cuts[1:]):

        value = quad(source.value, left, right, epsabs=1e-10, epsrel=1e-12)[0]

        moment = quad(lambda t:t*source.value(t), left, right, epsabs=1e-10, epsrel=1e-12)[0]

        assert source.integral(left,right) == pytest.approx(value, rel=2e-12, abs=1e-10)

        assert first_moment(source,left,right) == pytest.approx(moment, rel=2e-12, abs=1e-10)

    integral = quad(lambda t:float(source.value(t))**2, 0.,3.,

        points=[t for t in source.events if 0<t<3], epsabs=1e-8, epsrel=1e-12)[0]

    assert unknown_L2_squared(source) == pytest.approx(integral, rel=2e-12)


@pytest.mark.parametrize('name,original', [('PG10','SF01-PG10'),('SB150','SF01-SB150'),

                                        ('EC04','SF03-EC04'),('EC06','SF03-EC06')])

def test_source_shapes_and_prehistory_scale_together(binding, name, original):

    original_source = build_truth(binding,original).source

    scaled = make_sources(binding,mass=250.)[name]

    times = np.linspace(-.5,3.,431)

    np.testing.assert_array_equal(scaled.value(times),2.5*original_source.value(times))

    assert scaled.integral(-.5,0.) == 2.5*original_source.integral(-.5,0.)


def test_two_pulse_support_mass_and_smooth_endpoint_behavior():

    jump, smooth = TwoPulse('jump'), TwoPulse('cosine')

    for center,width,fraction in jump.components:

        left,right = center-width/2,center+width/2

        assert jump.integral(left,right) == pytest.approx(100*fraction)

        assert smooth.integral(left,right) == pytest.approx(100*fraction)

        assert smooth.value(center) == 2*jump.value(center)

        assert smooth.value(left) == smooth.value(right) == 0.

        epsilon = 1e-6

        assert abs(smooth.value(left+epsilon)/epsilon) < .1

    assert jump.integral(-.5,0.) == smooth.integral(-.5,0.) == 0.


@pytest.mark.parametrize('shape,mass', [('other',100),('jump',0),('cosine',-1),('jump',np.inf),('jump',True)])

def test_invalid_generators_fail(shape,mass):

    with pytest.raises(ValueError):

        TwoPulse(shape,mass)
