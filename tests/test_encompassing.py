"""Independent HAC regressions, sandwich calculations and encompassing restrictions."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, chi2
from statsmodels.regression.linear_model import OLS
from forecasting_tools import encompassing_test


def sample(n=120):
    rng = np.random.default_rng(719)
    f1, f2 = rng.normal(size=(2, n))
    shocks = rng.normal(size=n)
    noise = np.zeros(n)
    for i in range(n):
        noise[i] = shocks[i] + (.45*noise[i-1] if i else 0)
    return .7 + .8*f1 + .3*f2 + noise, f1, f2


@pytest.mark.parametrize('method,intercept', [('difference',False),('difference',True),
    ('cross_forecast',False),('cross_forecast',True),('unrestricted',False)])
@pytest.mark.parametrize('direction', ['1_encompasses_2','2_encompasses_1'])
@pytest.mark.parametrize('lag', [None,0,5,119])
def test_independent_regression_and_sandwich(method, intercept, direction, lag):
    y, f1, f2 = sample()
    n = len(y); first = direction == '1_encompasses_2'
    q = int(np.floor(4*(n/100)**(2/9))) if lag is None else lag
    if method == 'unrestricted':
        x = np.column_stack([np.ones(n), f1, f2]); target = y
        labels = ['intercept','forecast1','forecast2']
        r = np.array([[0,1,0],[0,0,1]])
        value = np.array([1,0]) if first else np.array([0,1])
    else:
        target = y-(f1 if first else f2)
        z = f1-f2 if method == 'difference' else (f2 if first else f1)
        x = np.column_stack([np.ones(n),z]) if intercept else z[:,None]
        labels = (['intercept'] if intercept else []) + ['forecast_difference' if method=='difference' else 'competing_forecast']
        r = np.zeros((1,x.shape[1])); r[0,-1]=1; value=np.zeros(1)
    fit = OLS(target,x).fit(cov_type='HAC',use_t=False,
        cov_kwds={'maxlags':q,'kernel':'bartlett','use_correction':False})
    result = encompassing_test(y,f1,f2,method=method,direction=direction,include_intercept=intercept,maxlags=lag,alpha=.1)
    beta = np.linalg.lstsq(x,target,rcond=None)[0]
    residual = target-x@beta; scores = x*residual[:,None]
    meat = scores.T@scores
    for j in range(1,q+1):
        cross = scores[j:].T@scores[:-j]
        meat += (1-j/(q+1))*(cross+cross.T)
    bread = np.linalg.inv(x.T@x); covariance=bread@meat@bread
    np.testing.assert_allclose(list(result['parameters'].values()),beta,atol=1e-12)
    np.testing.assert_allclose(list(result['parameters'].values()),fit.params,atol=1e-12)
    np.testing.assert_allclose(list(result['standard_errors'].values()),np.sqrt(np.diag(covariance)),rtol=1e-11)
    np.testing.assert_allclose(list(result['standard_errors'].values()),fit.bse,rtol=1e-12)
    delta=r@beta-value
    if method=='unrestricted':
        statistic=float(delta@np.linalg.solve(r@covariance@r.T,delta))
        p=chi2.sf(statistic,2)
        independent=fit.wald_test((r,value),use_f=False,scalar=True)
    else:
        statistic=float(delta[0]/np.sqrt((r@covariance@r.T)[0,0])); p=2*norm.sf(abs(statistic))
        independent=fit.t_test((r,value),use_t=False)
    assert result['statistic']==pytest.approx(statistic,rel=1e-10)
    assert result['statistic']==pytest.approx(float(np.asarray(independent.statistic).item()),rel=1e-12)
    assert result['pvalue']==pytest.approx(p,abs=1e-14)
    assert result['reject_null']==(p<.1)
    assert result['maxlags']==q
    assert result['include_intercept']==(intercept or method=='unrestricted')
    assert result['df']==(2 if method=='unrestricted' else None)
    assert result['distribution']==('chi2' if method=='unrestricted' else 'normal')
    assert result['df_resid']==n-x.shape[1]
    assert list(result['parameters'])==labels
    expected=np.column_stack([beta-norm.isf(.05)*fit.bse,beta+norm.isf(.05)*fit.bse])
    np.testing.assert_allclose(list(result['confidence_intervals'].values()),expected,atol=1e-12)


@pytest.mark.parametrize('method', ['difference','cross_forecast','unrestricted'])
def test_reverse_direction_by_swapping_forecasts(method):
    args=sample(); y,f1,f2=args
    first=encompassing_test(*args,method=method)
    swapped=encompassing_test(y,f2,f1,method=method,direction='2_encompasses_1')
    assert swapped['pvalue']==pytest.approx(first['pvalue'])
    assert swapped['statistic']==pytest.approx(-first['statistic'] if method=='difference' else first['statistic'])


@pytest.mark.parametrize('convert', [list,np.array,pd.Series])
def test_input_types(convert):
    args=sample()
    assert encompassing_test(*(convert(x) for x in args))==encompassing_test(*args)


@pytest.mark.parametrize('options', [
    {'method':'bad'},{'method':None},{'direction':'bad'},{'direction':[]},
    {'include_intercept':1},{'include_intercept':'yes'},
    {'method':'unrestricted','include_intercept':True},
    {'maxlags':-1},{'maxlags':120},{'maxlags':.5},{'maxlags':True},
    {'alpha':0},{'alpha':1},{'alpha':np.nan},{'alpha':np.inf},{'alpha':True},
])
def test_invalid_options(options):
    with pytest.raises(ValueError): encompassing_test(*sample(),**options)


@pytest.mark.parametrize('position',[0,1,2])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,'bad',True,1j])
def test_invalid_observations(position,value):
    args=[list(x) for x in sample()]; args[position][0]=value
    with pytest.raises((ValueError,TypeError)): encompassing_test(*args)


def test_lengths_indexes_masks_dimensions():
    args=sample()
    for i in range(3):
        short=list(args); short[i]=short[i][:-1]
        with pytest.raises(ValueError): encompassing_test(*short)
        labels=[pd.Series(x) for x in args]; labels[i].index=labels[i].index[::-1]
        with pytest.raises(ValueError): encompassing_test(*labels)
    with pytest.raises(ValueError): encompassing_test(args[0],pd.Series(args[1]),pd.Series(args[2],index=np.arange(1,121)))
    with pytest.raises(ValueError): encompassing_test([],[],[])
    with pytest.raises(ValueError): encompassing_test([[1,2]],[1,2],[2,3])
    masked=np.ma.array(args[0],mask=np.arange(120)==0)
    with pytest.raises(ValueError): encompassing_test(masked,args[1],args[2])


@pytest.mark.parametrize('method,intercept,n', [('difference',False,1),('difference',True,2),
    ('cross_forecast',False,1),('cross_forecast',True,2),('unrestricted',False,3)])
def test_insufficient_sample(method,intercept,n):
    with pytest.raises(ValueError,match='more observations'):
        encompassing_test(*sample(n),method=method,include_intercept=intercept)


def test_identical_and_collinear_forecasts():
    y,f1,f2=sample()
    for method in ['difference','unrestricted']:
        with pytest.raises(ValueError,match='rank-deficient'):
            encompassing_test(y,f1,f1,method=method)
    # Cross-forecast remains an identified orthogonality diagnostic.
    assert np.isfinite(encompassing_test(y,f1,f1,method='cross_forecast')['statistic'])
    with pytest.raises(ValueError,match='rank-deficient'):
        encompassing_test(y,f1,2*f1+1,method='unrestricted')


def test_constant_regressor_intercept_distinction():
    y,f1,f2=sample(); constant=np.ones(len(y))
    with pytest.raises(ValueError,match='rank-deficient'):
        encompassing_test(y,f1,constant,method='cross_forecast',include_intercept=True)
    with pytest.raises(ValueError,match='rank-deficient'):
        encompassing_test(y,f1,np.zeros(len(y)),method='cross_forecast')
    result=encompassing_test(y,f1,constant,method='cross_forecast')
    assert result['parameters']['competing_forecast']==pytest.approx(np.mean(y-f1))
    with pytest.raises(ValueError,match='rank-deficient'):
        encompassing_test(y,f1,f1-1,include_intercept=True)
    assert np.isfinite(encompassing_test(y,f1,f1-1)['statistic'])


@pytest.mark.parametrize('method', ['difference','cross_forecast','unrestricted'])
def test_perfect_fit(method):
    _,f1,f2=sample()
    y=f1+.4*(f1-f2) if method=='difference' else f1+.4*f2
    with pytest.raises(ValueError,match='residual variance'):
        encompassing_test(y,f1,f2,method=method)


def test_singular_and_indefinite_covariance():
    for matrix,message in [(np.zeros((2,2)),'singular'),(np.diag([1.,-1.]),'positive definite')]:
        with patch('statsmodels.regression.linear_model.OLSResults.cov_params',return_value=matrix):
            with pytest.raises(ValueError,match=message):
                encompassing_test(*sample(),include_intercept=True)


def test_overflow():
    with pytest.raises(ValueError,match='numerically invalid'):
        encompassing_test([1e308,-1e308,0],[1e308,1e308,1],[-1e308,-1e308,2])


def test_methods_are_not_interchangeable():
    args=sample()
    difference=encompassing_test(*args,include_intercept=True)
    cross=encompassing_test(*args,method='cross_forecast',include_intercept=True)
    assert abs(difference['statistic']-cross['statistic'])>.1
