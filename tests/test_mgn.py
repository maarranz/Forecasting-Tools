"""Independent classical-correlation and Bartlett HAC validation for MGN."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from forecasting_tools import mgn_test


def sample(n=120):
    rng = np.random.default_rng(517)
    v = rng.normal(size=n)
    innovations = rng.normal(size=n)
    residual = np.zeros(n)
    for i in range(n):
        residual[i] = innovations[i] + (.4*residual[i-1] if i else 0)
    u = .4 + .6*v + residual
    actual = 100 + rng.normal(size=n)
    return actual, actual-(u+v)/2, actual-(u-v)/2


def manual_regression(actual, forecast1, forecast2, lag=None):
    e1, e2 = np.asarray(actual)-forecast1, np.asarray(actual)-forecast2
    u, v = e1+e2, e1-e2
    x = np.column_stack([np.ones(len(u)), v])
    beta = np.linalg.lstsq(x, u, rcond=None)[0]
    residual = u-x@beta
    bread = np.linalg.inv(x.T@x)
    if lag is None:
        covariance = bread*(residual@residual)/(len(u)-2)
    else:
        scores = x*residual[:,None]
        meat = scores.T@scores
        for j in range(1,lag+1):
            cross = scores[j:].T@scores[:-j]
            meat += (1-j/(lag+1))*(cross+cross.T)
        covariance = bread@meat@bread
    return beta,covariance,u,v


@pytest.mark.parametrize('cov_type', ['nonrobust','HAC'])
def test_independent_estimates_covariance_statistic_and_intervals(cov_type):
    inputs=sample()
    lag=4 if cov_type=='HAC' else None
    result=mgn_test(*inputs,cov_type=cov_type,maxlags=lag,alpha=.1)
    beta,covariance,u,v=manual_regression(*inputs,lag=lag)
    se=np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(list(result['parameters'].values()),beta,atol=1e-12)
    np.testing.assert_allclose(list(result['standard_errors'].values()),se,rtol=1e-10)
    statistic=beta[1]/se[1]
    pvalue=2*t.sf(abs(statistic),len(u)-2) if cov_type=='nonrobust' else 2*norm.sf(abs(statistic))
    assert result['statistic']==pytest.approx(statistic,rel=1e-10)
    assert result['pvalue']==pytest.approx(pvalue,rel=1e-10)
    assert result['reject_null']==(pvalue<.1)
    critical=t.isf(.05,len(u)-2) if cov_type=='nonrobust' else norm.isf(.05)
    expected=np.column_stack([beta-critical*se,beta+critical*se])
    np.testing.assert_allclose(list(result['confidence_intervals'].values()),expected,rtol=1e-10)
    assert result['df']==(len(u)-2 if cov_type=='nonrobust' else None)
    assert result['df_resid']==len(u)-2
    assert result['maxlags']==lag
    assert result['distribution']==('t' if cov_type=='nonrobust' else 'normal')


def test_classical_correlation_identity():
    inputs=sample()
    result=mgn_test(*inputs,cov_type='nonrobust')
    _,_,u,v=manual_regression(*inputs)
    centered_u=u-u.mean()
    centered_v=v-v.mean()
    r=(centered_u@centered_v)/np.sqrt((centered_u@centered_u)*(centered_v@centered_v))
    statistic=r*np.sqrt((len(u)-2)/(1-r*r))
    assert result['correlation']==pytest.approx(r,rel=1e-12)
    assert result['statistic']==pytest.approx(statistic,rel=1e-12)
    assert result['pvalue']==pytest.approx(2*t.sf(abs(statistic),len(u)-2),rel=1e-12)


@pytest.mark.parametrize('n',[3,8,120])
def test_automatic_bandwidth(n):
    result=mgn_test(*sample(n))
    assert result['maxlags']==min(n-1,int(np.floor(4*(n/100)**(2/9))))


def test_zero_hac_bandwidth_and_integer_type():
    inputs=sample()
    result=mgn_test(*inputs,maxlags=np.int64(0))
    _,covariance,_,_=manual_regression(*inputs,lag=0)
    assert result['standard_errors']['slope']==pytest.approx(np.sqrt(covariance[1,1]))
    assert result['maxlags']==0


def test_msfe_difference_is_not_used_as_null():
    # Centered u/v covariance is exactly zero, but different mean errors give
    # sample MSFE difference 20. An intercept-only loss test would not match.
    v=np.arange(-2.,3.)+2
    u=10+np.array([1.,-2.,2.,-2.,1.])
    actual=np.zeros(5)
    result=mgn_test(actual,-(u+v)/2,-(u-v)/2,cov_type='nonrobust')
    assert result['parameters']['slope']==pytest.approx(0,abs=1e-14)
    assert result['pvalue']==pytest.approx(1)
    assert result['mse_difference']==pytest.approx(20)
    assert result['mean_errors']==pytest.approx({'model1':6,'model2':4})
    assert not result['reject_null']
    assert 'zero population means' in result['msfe_interpretation_note']


@pytest.mark.parametrize('cov_type',['HAC','nonrobust'])
def test_model_swap_and_error_scaling(cov_type):
    actual,first,second=sample()
    result=mgn_test(actual,first,second,cov_type=cov_type)
    swapped=mgn_test(actual,second,first,cov_type=cov_type)
    scaled=mgn_test(3*actual,3*first,3*second,cov_type=cov_type)
    assert swapped['statistic']==pytest.approx(-result['statistic'])
    assert swapped['correlation']==pytest.approx(-result['correlation'])
    assert swapped['pvalue']==pytest.approx(result['pvalue'])
    assert swapped['mse_difference']==pytest.approx(-result['mse_difference'])
    assert scaled['parameters']['slope']==pytest.approx(result['parameters']['slope'])
    assert scaled['statistic']==pytest.approx(result['statistic'])
    assert scaled['mse_difference']==pytest.approx(9*result['mse_difference'])


@pytest.mark.parametrize('constructor',[np.array,list,tuple,pd.Series])
def test_input_formats_and_immutability(constructor):
    inputs=sample()
    converted=[constructor(values) for values in inputs]
    result=mgn_test(*converted)
    assert result['nobs']==120
    for original,current in zip(inputs,converted):
        np.testing.assert_array_equal(original,current)


def test_pandas_indexes_for_all_pairs_and_mixed_inputs():
    inputs=sample(8)
    index=pd.date_range('2026-01-01',periods=8)
    series=[pd.Series(values,index=index,name=str(i)) for i,values in enumerate(inputs)]
    expected=mgn_test(*inputs)
    assert mgn_test(*series)['statistic']==pytest.approx(expected['statistic'])
    for i in range(3):
        changed=series.copy()
        changed[i]=changed[i].iloc[::-1]
        with pytest.raises(ValueError,match='indexes.*same order'):
            mgn_test(*changed)
    with pytest.raises(ValueError,match='indexes'):
        mgn_test(inputs[0],series[1],series[2].iloc[::-1])
    assert mgn_test(inputs[0],series[1],series[2])['statistic']==pytest.approx(expected['statistic'])


@pytest.mark.parametrize('bad',[[],[1],[1,2],[[1,2,3]],['1','2','3'],[True,False,True],
    [1,np.nan,2],[1,np.inf,2],[None,1,2],[1+1j,2,3],
    pd.Series([1,pd.NA,3],dtype='Float64'),pd.DataFrame({'x':[1,2,3]}),
    np.ma.array([1,2,3],mask=[0,1,0])])
def test_invalid_inputs_in_all_positions(bad):
    inputs=sample(3)
    for i in range(3):
        changed=list(inputs)
        changed[i]=bad
        with pytest.raises((ValueError,TypeError)):
            mgn_test(*changed)


@pytest.mark.parametrize('options',[{'cov_type':'HC0'},{'cov_type':None},{'cov_type':[]},
    *[{'alpha':v} for v in [0,1,-1,np.nan,np.inf,True,None,'0.05']],
    *[{'maxlags':v} for v in [-1,1.5,True,np.nan,np.inf,'1',120]],
    {'cov_type':'nonrobust','maxlags':0}])
def test_invalid_inference_options(options):
    with pytest.raises(ValueError):
        mgn_test(*sample(),**options)


@pytest.mark.parametrize('cov_type',['HAC','nonrobust'])
def test_constant_sums_differences_and_collinear_errors(cov_type):
    varying=np.arange(1.,9.)
    for first,second in [(varying,varying),(varying,10-varying),(varying,2*varying)]:
        with pytest.raises(ValueError,match='variance|collinear|degenerate'):
            mgn_test(np.zeros(8),-first,-second,cov_type=cov_type)


def test_near_perfect_collinearity():
    v=np.arange(10.)
    u=2+.5*v+np.array([1,-1]*5)*1e-8
    with pytest.raises(ValueError,match='near-perfectly collinear'):
        mgn_test(np.zeros(10),-(u+v)/2,-(u-v)/2)


def test_numerically_rank_deficient_design():
    v=1e8+np.arange(8)*1e-6
    u=np.array([1.,2.,4.,3.,5.,8.,6.,7.])
    with pytest.raises(ValueError,match='degenerate|rank-deficient'):
        mgn_test(np.zeros(8),-(u+v)/2,-(u-v)/2)


def test_singular_hac_covariance():
    u=np.array([0.,2.,4.,4.])
    v=np.array([0.,0.,1.,1.])
    with pytest.raises(ValueError,match='covariance'):
        mgn_test(np.zeros(4),-(u+v)/2,-(u-v)/2,maxlags=0)


def test_overflow_is_informative():
    with pytest.raises(ValueError,match='overflow|nonfinite|numerically invalid'):
        mgn_test([1e308,-1e308,1e308],[-1e308,1e308,0],[0,0,0])


def test_invalid_covariance_is_informative():
    from statsmodels.regression.linear_model import RegressionResults
    with patch.object(RegressionResults,'cov_params',return_value=np.full((2,2),np.nan)):
        with pytest.raises(ValueError,match='covariance.*nonfinite'):
            mgn_test(*sample())
    with patch.object(RegressionResults,'cov_params',return_value=np.diag([1.,-1.])):
        with pytest.raises(ValueError,match='not positive definite'):
            mgn_test(*sample())


def test_return_structure_and_descriptive_values():
    inputs=sample()
    result=mgn_test(*inputs)
    assert set(result)=={'method','null_hypothesis','nobs','statistic','distribution','df',
        'df_resid','n_restrictions','pvalue','alpha','reject_null','cov_type','maxlags',
        'correlation','parameters','standard_errors','confidence_intervals',
        'mean_errors','mse','mse_difference','msfe_interpretation_note'}
    actual,first,second=inputs
    e1,e2=actual-first,actual-second
    assert result['mean_errors']==pytest.approx({'model1':e1.mean(),'model2':e2.mean()})
    assert result['mse']==pytest.approx({'model1':np.mean(e1**2),'model2':np.mean(e2**2)})
    assert result['mse_difference']==pytest.approx(np.mean(e1**2)-np.mean(e2**2))
    assert result['method']=='mgn' and result['null_hypothesis']=='slope = 0'
    assert result['n_restrictions']==1
    assert isinstance(result['reject_null'],bool)
    assert result['parameters'].keys()==result['standard_errors'].keys()==result['confidence_intervals'].keys()
