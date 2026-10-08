"""Direct-formula and independent Statsmodels HAC validation of DM."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t
from statsmodels.regression.linear_model import OLS
from forecasting_tools import dm_test


def sample(n=90):
    rng = np.random.default_rng(6712)
    shocks = rng.normal(size=n)
    e1 = np.zeros(n)
    for i in range(n):
        e1[i] = .6 + shocks[i] + (.6*e1[i-1] if i else 0)
    e2 = rng.normal(size=n)
    actual = np.full(n, 100.)
    return actual, actual-e1, actual-e2


@pytest.mark.parametrize('loss', ['squared', 'absolute'])
@pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
@pytest.mark.parametrize('lag', [None, 0, 4, 89])
@pytest.mark.parametrize('hln', [False, True])
def test_independent_hac_regression(loss, alternative, lag, hln):
    y, f1, f2 = sample()
    transform = np.square if loss == 'squared' else np.abs
    l1, l2 = transform(y-f1), transform(y-f2)
    d = l1-l2
    n, h = len(d), 3
    q = min(n-1, int(np.floor(4*(n/100)**(2/9)))) if lag is None else lag
    fit = OLS(d, np.ones((n, 1))).fit(cov_type='HAC', use_t=False,
        cov_kwds={'maxlags': q, 'kernel': 'bartlett', 'use_correction': False})
    result = dm_test(y, f1, f2, loss=loss, alternative=alternative,
                     maxlags=lag, h=h, hln=hln)
    assert result['mean_loss_diff'] == pytest.approx(fit.params[0])
    assert result['se'] == pytest.approx(fit.bse[0], rel=1e-12)
    assert result['dm_statistic'] == pytest.approx(fit.tvalues[0], rel=1e-12)
    assert result['long_run_variance'] == pytest.approx(n*fit.bse[0]**2)
    assert result['mean_loss1'] == pytest.approx(l1.mean())
    assert result['mean_loss2'] == pytest.approx(l2.mean())
    factor = np.sqrt((n+1-2*h+h*(h-1)/n)/n) if hln else None
    stat = fit.tvalues[0]*(factor if hln else 1)
    ref = t(n-1) if hln else norm
    p = 2*ref.sf(abs(stat)) if alternative == 'two-sided' else ref.cdf(stat) if alternative == 'less' else ref.sf(stat)
    assert result['statistic'] == pytest.approx(stat)
    assert result['pvalue'] == pytest.approx(p)
    assert result['df'] == (n-1 if hln else None)
    assert result['distribution'] == ('t' if hln else 'normal')
    assert result['hln_factor'] == pytest.approx(factor) if hln else result['hln_factor'] is None
    assert result['maxlags'] == q
    assert result['reject_null'] == (p < .05)
    swapped = dm_test(y, f2, f1, loss=loss, alternative=alternative, maxlags=lag, h=h, hln=hln)
    assert swapped['statistic'] == pytest.approx(-result['statistic'])
    assert swapped['pvalue'] == pytest.approx(p if alternative == 'two-sided' else 1-p)
    plain = dm_test(y, f1, f2, loss=loss, maxlags=lag, h=h)
    assert plain['dm_statistic'] == result['dm_statistic']
    assert plain['long_run_variance'] == result['long_run_variance']
    assert plain['se'] == result['se']


@pytest.mark.parametrize('loss', ['squared', 'absolute'])
@pytest.mark.parametrize('hln', [False, True])
def test_hand_calculation_short_sample(loss, hln):
    # Errors (1, 2, 3) versus (0, 0, 0): independent q=1 calculation.
    d = np.array([1., 4., 9.]) if loss == 'squared' else np.array([1., 2., 3.])
    mean = sum(d)/3
    centered = [x-mean for x in d]
    omega = sum(x*x for x in centered)/3 + sum(centered[i]*centered[i-1] for i in [1, 2])/3
    r = dm_test([0., 0., 0.], [-1., -2., -3.], [0., 0., 0.], loss=loss, maxlags=1, hln=hln)
    assert r['long_run_variance'] == pytest.approx(omega)
    assert r['dm_statistic'] == pytest.approx(mean/np.sqrt(omega/3))
    assert dm_test([0, 0], [-1, -2], [0, 0], loss=loss, hln=hln)['n'] == 2


@pytest.mark.parametrize('inputs', [list, np.array, pd.Series])
def test_input_types(inputs):
    data = sample()
    assert dm_test(*(inputs(x) for x in data)) == dm_test(*data)


@pytest.mark.parametrize('options', [
    {'loss': 'bad'}, {'loss': None}, {'alternative': 'bad'}, {'alternative': []},
    {'h': 0}, {'h': -1}, {'h': 1.5}, {'h': True},
    {'alpha': 0}, {'alpha': 1}, {'alpha': np.nan}, {'alpha': np.inf}, {'alpha': True},
    {'maxlags': -1}, {'maxlags': 90}, {'maxlags': 1.5}, {'maxlags': True},
    {'hln': 1}, {'hln': 'yes'}, {'hln': True, 'h': 90}, {'hln': True, 'h': 91},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        dm_test(*sample(), **options)


@pytest.mark.parametrize('position', [0, 1, 2])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 'bad', True, 1j])
def test_invalid_observations(position, value):
    data = [list(x) for x in sample()]
    data[position][2] = value
    with pytest.raises((ValueError, TypeError)):
        dm_test(*data)


def test_lengths_indexes_and_dimensions():
    data = sample()
    for i in range(3):
        short = list(data); short[i] = short[i][:-1]
        with pytest.raises(ValueError): dm_test(*short)
        indexed = [pd.Series(x) for x in data]
        indexed[i].index = indexed[i].index[::-1]
        with pytest.raises(ValueError): dm_test(*indexed)
    with pytest.raises(ValueError): dm_test(data[0], pd.Series(data[1]), pd.Series(data[2], index=np.arange(1,91)))
    with pytest.raises(ValueError): dm_test([1], [2], [3])
    with pytest.raises(ValueError): dm_test([], [], [])
    with pytest.raises(ValueError): dm_test([[1,2]], [1,2], [2,3])


@pytest.mark.parametrize('f1,f2', [([1,1,1], [1,1,1]), ([1,1,1], [2,2,2]),
                                 ([1e200,2e200,3e200], [0,0,0]),
                                 ([1e-200,2e-200,3e-200], [0,0,0])])
def test_degenerate_or_overflow(f1, f2):
    with pytest.raises(ValueError): dm_test([0,0,0], f1, f2)


def test_horizon_does_not_override_bandwidth_or_shift_data():
    args = sample()
    baseline = dm_test(*args, maxlags=0)
    longer = dm_test(*args, h=6, maxlags=0)
    assert longer['maxlags'] == 0
    assert longer['statistic'] == baseline['statistic']
    auto = dm_test(*args, h=6)
    assert auto['maxlags'] == 3  # Less than h-1 is allowed by design.


def test_scaled_variation_near_constant_rejected():
    with pytest.raises(ValueError, match='degenerate'):
        dm_test([0,0,0], [-1, -1, -np.nextafter(1., 2.)], [0,0,0])
