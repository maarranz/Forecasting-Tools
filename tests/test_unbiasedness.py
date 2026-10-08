"""Independent regression and Bartlett sandwich checks for unbiasedness."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, f, norm, t

from forecasting_tools import unbiasedness_test


def manual_fit(actual, forecast, joint, lags=None):
    n = len(actual)
    x = np.column_stack([np.ones(n), forecast]) if joint else np.ones((n, 1))
    y = actual if joint else actual - forecast
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    if lags is None:
        covariance = bread * (residuals @ residuals) / (n - x.shape[1])
    else:
        scores = x * residuals[:, None]
        meat = scores.T @ scores
        for lag in range(1, lags + 1):
            cross = scores[lag:].T @ scores[:-lag]
            meat += (1 - lag / (lags + 1)) * (cross + cross.T)
        covariance = bread @ meat @ bread
    return beta, covariance


@pytest.fixture
def sample():
    rng = np.random.default_rng(315)
    forecast = 10 + rng.normal(size=80)
    innovations = rng.normal(size=80)
    errors = np.zeros(80)
    for i in range(1, 80):
        errors[i] = .4 * errors[i-1] + innovations[i]
    return .3 + 1.05 * forecast + errors, forecast


@pytest.mark.parametrize('method', ['mean_error', 'mincer_zarnowitz'])
@pytest.mark.parametrize('cov_type', ['nonrobust', 'HAC'])
def test_independent_regression_covariance_statistic_and_intervals(sample, method, cov_type):
    actual, forecast = sample
    joint = method == 'mincer_zarnowitz'
    lag = 3 if cov_type == 'HAC' else None
    result = unbiasedness_test(actual, forecast, method=method, cov_type=cov_type,
                               maxlags=lag, alpha=.1)
    beta, covariance = manual_fit(actual, forecast, joint, lag)
    np.testing.assert_allclose(list(result['parameters'].values()), beta, atol=1e-12)
    np.testing.assert_allclose(list(result['standard_errors'].values()), np.sqrt(np.diag(covariance)), rtol=1e-10)
    k = 2 if joint else 1
    if joint:
        difference = beta - np.array([0, 1])
        wald = difference @ np.linalg.solve(covariance, difference)
        statistic = wald / 2 if cov_type == 'nonrobust' else wald
        pvalue = f.sf(statistic, 2, len(actual)-2) if cov_type == 'nonrobust' else chi2.sf(statistic, 2)
        assert result['distribution'] == ('F' if cov_type == 'nonrobust' else 'chi2')
    else:
        statistic = beta[0] / np.sqrt(covariance[0, 0])
        pvalue = 2*t.sf(abs(statistic), len(actual)-1) if cov_type == 'nonrobust' else 2*norm.sf(abs(statistic))
        assert result['distribution'] == ('t' if cov_type == 'nonrobust' else 'normal')
    np.testing.assert_allclose(result['statistic'], statistic, rtol=1e-10)
    np.testing.assert_allclose(result['pvalue'], pvalue, rtol=1e-10)
    critical = t.isf(.05, len(actual)-k) if cov_type == 'nonrobust' else norm.isf(.05)
    expected_ci = np.column_stack([beta-critical*np.sqrt(np.diag(covariance)), beta+critical*np.sqrt(np.diag(covariance))])
    np.testing.assert_allclose(list(result['confidence_intervals'].values()), expected_ci, rtol=1e-10)
    assert result['reject_null'] == (pvalue < .1)
    assert result['df_resid'] == len(actual)-k
    assert result['n_restrictions'] == k
    assert result['maxlags'] == lag
    assert result['df_num'] == (2 if joint and cov_type=='nonrobust' else None)
    assert result['df_denom'] == (len(actual)-2 if joint and cov_type=='nonrobust' else None)
    assert result['df'] == (len(actual)-1 if not joint and cov_type=='nonrobust' else 2 if joint and cov_type=='HAC' else None)


def test_hand_calculated_mean_error_and_bias():
    # Errors 2,-3,4: mean=1, sample variance=13, SE=sqrt(13/3).
    result = unbiasedness_test([10, 20, 30], [8, 23, 26], cov_type='nonrobust')
    assert result['parameters']['intercept'] == pytest.approx(1)
    assert result['standard_errors']['intercept'] == pytest.approx(np.sqrt(13/3))
    assert result['statistic'] == pytest.approx(np.sqrt(3/13))
    assert result['pvalue'] == pytest.approx(2*t.sf(np.sqrt(3/13), 2))
    # Deterministic mean-zero errors with nonzero variance.
    forecast = np.arange(1., 9.)
    errors = np.array([-1., 1., -2., 2., -1., 1., -2., 2.])
    zero = unbiasedness_test(forecast+errors, forecast, cov_type='nonrobust')
    assert zero['parameters']['intercept'] == pytest.approx(0, abs=1e-14)
    assert zero['pvalue'] == pytest.approx(1)
    assert not zero['reject_null']
    biased = unbiasedness_test(forecast+errors+10, forecast, cov_type='nonrobust')
    assert biased['parameters']['intercept'] == pytest.approx(10)
    assert biased['reject_null']


@pytest.mark.parametrize('n', [2, 8, 80])
def test_automatic_bandwidth(n):
    actual = np.arange(n, dtype=float)
    forecast = actual + np.arange(n, dtype=float)**2 + 1
    result = unbiasedness_test(actual, forecast)
    assert result['maxlags'] == min(n-1, int(np.floor(4*(n/100)**(2/9))))


@pytest.mark.parametrize('method', ['mean_error', 'mincer_zarnowitz'])
def test_hac_zero_lags_and_numpy_integer(sample, method):
    actual, forecast = sample
    result = unbiasedness_test(actual, forecast, method=method, maxlags=np.int64(0))
    _, covariance = manual_fit(actual, forecast, method=='mincer_zarnowitz', 0)
    np.testing.assert_allclose(list(result['standard_errors'].values()), np.sqrt(np.diag(covariance)), rtol=1e-10)
    assert result['maxlags'] == 0


@pytest.mark.parametrize('constructor', [np.array, list, tuple, pd.Series])
def test_input_types_and_immutability(sample, constructor):
    actual, forecast = sample
    a, b = constructor(actual), constructor(forecast)
    result = unbiasedness_test(a, b)
    assert result['parameters']['intercept'] == pytest.approx(np.mean(actual-forecast))
    np.testing.assert_array_equal(a, actual)
    np.testing.assert_array_equal(b, forecast)


@pytest.mark.parametrize('method', ['mean_error', 'mincer_zarnowitz'])
@pytest.mark.parametrize('cov_type', ['HAC', 'nonrobust'])
def test_structure(sample, method, cov_type):
    result = unbiasedness_test(*sample, method=method, cov_type=cov_type)
    assert set(result) == {'method','null_hypothesis','statistic','distribution','pvalue','alpha',
                           'reject_null','nobs','cov_type','maxlags','parameters','standard_errors',
                           'confidence_intervals','n_restrictions','df_resid','df','df_num','df_denom'}
    assert isinstance(result['reject_null'], bool)
    assert isinstance(result['nobs'], int)
    assert 0 <= result['pvalue'] <= 1
    assert result['method'] == method and result['cov_type'] == cov_type
    assert result['alpha'] == .05
    assert set(result['parameters']) == ({'intercept','slope'} if method=='mincer_zarnowitz' else {'intercept'})
    assert result['parameters'].keys() == result['standard_errors'].keys() == result['confidence_intervals'].keys()
    for key, (lower, upper) in result['confidence_intervals'].items():
        assert isinstance(lower,float) and lower < result['parameters'][key] < upper


@pytest.mark.parametrize('bad', [[], [1], [np.nan, 2, 3], [np.inf, 2, 3],
                                [None, 2, 3], pd.Series([1,pd.NA,3],dtype='Float64'),
                                [[1,2,3]], ['1','2','3'], [True, False, True],
                                [1+1j,2,3], np.ma.array([1,2,3],mask=[0,1,0])])
def test_invalid_samples(bad):
    for a, b in [(bad,[1,2,3]), ([1,2,3],bad)]:
        with pytest.raises((ValueError,TypeError)):
            unbiasedness_test(a,b)


def test_mismatched_lengths_and_series_indexes():
    with pytest.raises(ValueError,match='length'):
        unbiasedness_test([1,2,3],[1,2])
    actual = pd.Series([1,2,3],index=['a','b','c'])
    with pytest.raises(ValueError,match='indexes.*same order'):
        unbiasedness_test(actual,pd.Series([1,2,3],index=['c','b','a']))
    with pytest.raises(ValueError,match='indexes'):
        unbiasedness_test(actual,pd.Series([1,2,3],index=['a','b','d']))
    # Mixed labeled/unlabeled inputs remain positional.
    assert unbiasedness_test(actual,[0,2,4])['nobs'] == 3


@pytest.mark.parametrize('options', [
    {'method':'other'}, {'method':None}, {'method':[]},
    {'cov_type':'HC1'}, {'cov_type':None}, {'cov_type':[]},
    *[{'alpha':v} for v in [0,1,-1,np.nan,np.inf,True,'0.05',None]],
    *[{'maxlags':v} for v in [-1,1.5,True,np.nan,np.inf,'1',80]],
    {'cov_type':'nonrobust','maxlags':0},
])
def test_invalid_inference_options(sample, options):
    with pytest.raises(ValueError):
        unbiasedness_test(*sample, **options)


@pytest.mark.parametrize('forecast', [np.ones(5), np.full(5,1e16), 1+np.arange(5)*1e-16])
def test_rank_deficient_mincer_zarnowitz(forecast):
    with pytest.raises(ValueError,match='rank-deficient'):
        unbiasedness_test(np.arange(5.),forecast,method='mincer_zarnowitz')


def test_insufficient_mincer_zarnowitz():
    with pytest.raises(ValueError,match='at least 3'):
        unbiasedness_test([1,2],[0,3],method='mincer_zarnowitz')


@pytest.mark.parametrize('method', ['mean_error','mincer_zarnowitz'])
def test_degenerate_inference_is_explained(method):
    with pytest.raises(ValueError,match='degenerate|variance|covariance'):
        unbiasedness_test([1,2,3,4],[1,2,3,4],method=method)
    with pytest.raises(ValueError,match='degenerate|variance|covariance'):
        unbiasedness_test([2,3,4,5],[1,2,3,4],method=method)


def test_zero_joint_restrictions_and_deliberate_calibration_bias():
    forecast = np.arange(1., 6.)
    errors = np.array([1., -2., 2., -2., 1.])
    result = unbiasedness_test(forecast+errors, forecast,
                               method='mincer_zarnowitz', cov_type='nonrobust')
    assert result['parameters']['intercept'] == pytest.approx(0, abs=1e-14)
    assert result['parameters']['slope'] == pytest.approx(1)
    assert result['statistic'] == pytest.approx(0, abs=1e-24)
    assert result['pvalue'] == pytest.approx(1)
    assert not result['reject_null']
    biased = unbiasedness_test(20+2*forecast+errors, forecast,
                               method='mincer_zarnowitz', cov_type='nonrobust')
    assert biased['parameters']['intercept'] == pytest.approx(20)
    assert biased['parameters']['slope'] == pytest.approx(2)
    assert biased['reject_null']


def test_singular_hac_covariance_is_explained():
    with pytest.raises(ValueError, match='covariance'):
        unbiasedness_test([0,2,4,4],[0,0,1,1],method='mincer_zarnowitz',maxlags=0)
