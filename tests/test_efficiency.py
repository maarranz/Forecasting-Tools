"""Independent portmanteau and regression/sandwich efficiency checks."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, f, norm, t
from statsmodels.tsa.arima.model import ARIMA

from forecasting_tools import orthogonality_test, weak_efficiency_test


def manual_ljung_box(values, lag, df):
    centered = values - np.mean(values)
    n = len(values)
    denominator = centered @ centered
    correlations = [centered[j:] @ centered[:-j] / denominator for j in range(1, lag+1)]
    statistic = n*(n+2)*sum(r*r/(n-j) for j, r in enumerate(correlations, 1))
    return statistic, chi2.sf(statistic, df)


def simulated_errors(kind, n=800):
    rng = np.random.default_rng(984)
    innovations = rng.normal(size=n+300)
    if kind == 'white':
        return innovations[300:]
    if kind == 'ma1':
        return (innovations[1:] + .45*innovations[:-1])[299:]
    if kind == 'ma2':
        return (innovations[2:] + .35*innovations[1:-1] - .2*innovations[:-2])[298:]
    correlated = np.zeros_like(innovations)
    for i in range(1,len(correlated)):
        correlated[i] = .8*correlated[i-1] + innovations[i]
    return correlated[300:]


@pytest.mark.parametrize('kind', ['white','ar'])
def test_one_step_ljung_box_independently(kind):
    errors = simulated_errors(kind)
    result = weak_efficiency_test(errors,np.zeros(len(errors)),lags=10,alpha=.1)
    statistic,pvalue = manual_ljung_box(errors,10,10)
    assert result['statistic'] == pytest.approx(statistic, rel=1e-12)
    assert result['pvalue'] == pytest.approx(pvalue, rel=1e-12)
    assert result['df']==10 and result['model_df']==0
    assert result['ma_coefficients']=={} and result['estimation'] is None
    assert result['reject_null']==(pvalue<.1)
    assert result['diagnostic_nobs']==len(errors)


def test_serial_dependence_affects_portmanteau_magnitude():
    white = simulated_errors('white')
    serial = simulated_errors('ar')
    qwhite = weak_efficiency_test(white,np.zeros(len(white)))['statistic']
    qserial = weak_efficiency_test(serial,np.zeros(len(serial)))['statistic']
    assert qserial > 10*qwhite


@pytest.mark.parametrize('h,kind', [(2,'ma1'),(3,'ma2'),(2,'ar')])
def test_multistep_fit_and_manual_residual_diagnostic(h,kind):
    errors = simulated_errors(kind)
    result = weak_efficiency_test(errors+1,np.zeros(len(errors)),h=h,lags=12)
    # Independent fit plus an explicit, non-Statsmodels portmanteau formula.
    reference = ARIMA(errors+1,order=(0,0,h-1),trend='c',enforce_invertibility=True).fit()
    residuals = reference.resid[int(reference.loglikelihood_burn):]
    statistic,pvalue = manual_ljung_box(residuals,12,12-(h-1))
    assert result['statistic']==pytest.approx(statistic,rel=1e-10)
    assert result['pvalue']==pytest.approx(pvalue,rel=1e-10,abs=1e-250)
    np.testing.assert_allclose(list(result['ma_coefficients'].values()),reference.maparams)
    assert result['df']==12-(h-1)
    assert result['model_df']==h-1
    assert result['estimation']['converged'] and result['estimation']['invertible']
    assert result['estimation']['mean']==pytest.approx(reference.params[0])
    assert result['estimation']['sigma2']==pytest.approx(reference.params[-1])
    assert result['estimation']['llf']==pytest.approx(reference.llf)
    assert result['estimation']['aic']==pytest.approx(reference.aic)
    assert result['estimation']['bic']==pytest.approx(reference.bic)
    assert result['diagnostic_nobs']==len(residuals)
    if kind=='ma1':
        assert result['ma_coefficients']['ma.L1']==pytest.approx(.45,abs=.15)
    if kind=='ma2':
        np.testing.assert_allclose(list(result['ma_coefficients'].values()),[.35,-.2],atol=.15)


def test_multistep_beyond_allowed_dependence():
    noise=simulated_errors('ma1')
    extra=simulated_errors('ar')
    compatible=weak_efficiency_test(noise,np.zeros(len(noise)),h=2,lags=12)
    incompatible=weak_efficiency_test(extra,np.zeros(len(extra)),h=2,lags=12)
    assert incompatible['statistic']>10*compatible['statistic']


def regression_sample(predictive=True):
    rng=np.random.default_rng(774)
    z=rng.normal(size=(100,2))
    x=np.column_stack([np.ones(100),z])
    noise=rng.normal(size=100)
    # Exact finite-sample orthogonality, not a random nonrejection assertion.
    residuals=noise-x@np.linalg.lstsq(x,noise,rcond=None)[0]
    beta=np.array([.5,.8,-.4]) if predictive else np.zeros(3)
    return x@beta+residuals,z,beta


@pytest.mark.parametrize('predictive',[False,True])
@pytest.mark.parametrize('cov_type',['nonrobust','HAC'])
def test_orthogonality_manual_inference(predictive,cov_type):
    errors,z,known_beta=regression_sample(predictive)
    information=pd.DataFrame(z,columns=['signal','lagged_information'])
    lag=4 if cov_type=='HAC' else None
    result=orthogonality_test(errors,np.zeros(len(errors)),information,h=2,
                              cov_type=cov_type,maxlags=lag,alpha=.1)
    x=np.column_stack([np.ones(len(errors)),z])
    beta=np.linalg.lstsq(x,errors,rcond=None)[0]
    residuals=errors-x@beta
    bread=np.linalg.inv(x.T@x)
    if cov_type=='nonrobust':
        covariance=bread*(residuals@residuals)/(len(errors)-3)
    else:
        scores=x*residuals[:,None]
        meat=scores.T@scores
        for j in range(1,lag+1):
            cross=scores[j:].T@scores[:-j]
            meat+=(1-j/(lag+1))*(cross+cross.T)
        covariance=bread@meat@bread
    np.testing.assert_allclose(list(result['parameters'].values()),known_beta,atol=1e-12)
    np.testing.assert_allclose(list(result['standard_errors'].values()),np.sqrt(np.diag(covariance)),rtol=1e-10)
    quadratic=beta@np.linalg.solve(covariance,beta)
    statistic=quadratic/3 if cov_type=='nonrobust' else quadratic
    pvalue=f.sf(statistic,3,97) if cov_type=='nonrobust' else chi2.sf(statistic,3)
    assert result['statistic']==pytest.approx(statistic,rel=1e-10,abs=1e-24)
    assert result['pvalue']==pytest.approx(pvalue,rel=1e-10)
    assert result['reject_null']==(pvalue<.1)
    assert result['n_restrictions']==3 and result['df_resid']==97
    assert result['distribution']==('F' if cov_type=='nonrobust' else 'chi2')
    assert result['df']==(None if cov_type=='nonrobust' else 3)
    assert result['df_num']==(3 if cov_type=='nonrobust' else None)
    assert result['df_denom']==(97 if cov_type=='nonrobust' else None)
    critical=t.isf(.05,97) if cov_type=='nonrobust' else norm.isf(.05)
    intervals=np.column_stack([beta-critical*np.sqrt(np.diag(covariance)),beta+critical*np.sqrt(np.diag(covariance))])
    np.testing.assert_allclose(list(result['confidence_intervals'].values()),intervals,rtol=1e-10,atol=1e-12)
    assert result['information_names']==['signal','lagged_information']
    assert result['maxlags']==lag
    assert result['bandwidth_covers_overlap']==(None if cov_type=='nonrobust' else True)


@pytest.mark.parametrize('constructor',[list,np.array,pd.Series])
def test_single_information_variable_and_automatic_bandwidth(constructor):
    errors,z,_=regression_sample()
    information=constructor(z[:,0])
    result=orthogonality_test(errors,np.zeros(100),information,h=6)
    assert result['maxlags']==4
    assert result['bandwidth_covers_overlap'] is False
    assert result['information_names']==['information_1']
    zero=orthogonality_test(errors,np.zeros(100),information,maxlags=np.int64(0))
    assert zero['maxlags']==0


def test_labels_positional_pairing_and_immutability():
    errors,z,_=regression_sample()
    index=pd.date_range('2026-01-01',periods=100)
    a=pd.Series(errors,index=index)
    b=pd.Series(np.zeros(100),index=index)
    info=pd.Series(z[:,0],index=index,name='origin_signal')
    before=info.copy()
    assert orthogonality_test(a,b,info)['information_names']==['origin_signal']
    assert orthogonality_test(errors,np.zeros(100),info)['nobs']==100
    pd.testing.assert_series_equal(info,before)
    for x,y in [(a,np.zeros(100)),(errors,b),(a,b)]:
        with pytest.raises(ValueError,match='indexes.*same order'):
            orthogonality_test(x,y,info.iloc[::-1])
    with pytest.raises(ValueError,match='indexes'):
        weak_efficiency_test(a,b.iloc[::-1])


@pytest.mark.parametrize('h',[0,-1,1.5,True,np.nan,None,'2'])
def test_invalid_horizons(h):
    with pytest.raises(ValueError,match='h must'):
        weak_efficiency_test([1,2,4,3],[0]*4,h=h,lags=2)
    with pytest.raises(ValueError,match='h must'):
        orthogonality_test([1,2,4,3],[0]*4,[0,1,0,1],h=h)


@pytest.mark.parametrize('alpha',[0,1,-1,np.nan,np.inf,True,'0.05',None])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError,match='alpha'):
        weak_efficiency_test([1,2,4,3],[0]*4,alpha=alpha,lags=2)
    with pytest.raises(ValueError,match='alpha'):
        orthogonality_test([1,2,4,3],[0]*4,[0,1,0,1],alpha=alpha)


@pytest.mark.parametrize('lags',[0,-1,1.5,True,None,[1,2],4])
def test_invalid_diagnostic_lags(lags):
    with pytest.raises(ValueError,match='lag'):
        weak_efficiency_test([1,2,4,3],[0]*4,lags=lags)


def test_insufficient_multistep_lag_and_sample():
    with pytest.raises(ValueError,match='exceed.*MA order'):
        weak_efficiency_test(np.arange(20),np.zeros(20),h=3,lags=2)
    with pytest.raises(ValueError,match='more than ma_order'):
        weak_efficiency_test([1,2,4],[0,0,0],h=2,lags=2)


@pytest.mark.parametrize('bad',[[],[1],[1,np.nan,2],[1,np.inf,2],[1,None,2],['1','2','3'],[True,False,True],[[1,2,3]],[1+1j,2,3]])
def test_invalid_forecast_inputs(bad):
    for a,b in [(bad,[0,0,0]),([1,2,3],bad)]:
        with pytest.raises((ValueError,TypeError)):
            weak_efficiency_test(a,b,lags=1)
        with pytest.raises((ValueError,TypeError)):
            orthogonality_test(a,b,[0,1,0])


@pytest.mark.parametrize('bad',[np.ones(10),np.ones((10,2)),np.empty((10,0)),np.ones((9,2)),
    np.ones((10,2,1)),np.array([np.nan]+[1]*9),np.array([np.inf]+[1]*9),
    pd.Series([1,pd.NA]+[1]*8,dtype='Float64'), ['1']*10,[True]*10,
    np.ma.array(np.arange(10),mask=[True]+[False]*9)])
def test_invalid_information(bad):
    with pytest.raises((ValueError,TypeError)):
        orthogonality_test(np.arange(10)**2,np.zeros(10),bad)


def test_regression_degeneracy_and_names():
    with pytest.raises(ValueError,match='more observations'):
        orthogonality_test([1,3],[0,0],[0,1])
    with pytest.raises(ValueError,match='rank-deficient'):
        orthogonality_test(np.arange(10)**2,np.zeros(10),np.column_stack([np.arange(10),2*np.arange(10)]))
    with pytest.raises(ValueError,match='variance'):
        orthogonality_test([1,2,3,4],[0]*4,[1,2,3,4])
    for info in [pd.Series(np.arange(10),name='intercept'),
                 pd.DataFrame(np.arange(20).reshape(10,2),columns=['x','x'])]:
        with pytest.raises(ValueError,match='names'):
            orthogonality_test(np.arange(10)**2,np.zeros(10),info)
    with pytest.raises(ValueError,match='covariance'):
        orthogonality_test([0,2,4,4],[0]*4,[0,0,1,1],maxlags=0)


@pytest.mark.parametrize('options',[{'cov_type':'HC0'},{'cov_type':None},
    *[{'maxlags':x} for x in [-1,1.5,True,np.nan,'1',100]],
    {'cov_type':'nonrobust','maxlags':0}])
def test_invalid_regression_inference(options):
    errors,z,_=regression_sample()
    with pytest.raises(ValueError):
        orthogonality_test(errors,np.zeros(100),z,**options)


def test_weak_degenerate_errors_and_ma_failures():
    with pytest.raises(ValueError,match='variance'):
        weak_efficiency_test(np.ones(20),np.zeros(20),lags=2)
    with patch.object(ARIMA,'fit',side_effect=np.linalg.LinAlgError('failure')):
        with pytest.raises(ValueError,match='estimation failed'):
            weak_efficiency_test(simulated_errors('ma1'),np.zeros(800),h=2)
    with patch.object(ARIMA,'fit',return_value=SimpleNamespace(mle_retvals={'converged':False})):
        with pytest.raises(ValueError,match='did not converge'):
            weak_efficiency_test(simulated_errors('ma1'),np.zeros(800),h=2)


def test_ma_nonfinite_and_degenerate_residuals():
    base=dict(mle_retvals={'converged':True},param_names=['const','ma.L1','sigma2'],
              params=np.array([0.,.3,1.]),bse=np.ones(3),maparams=np.array([.3]),
              loglikelihood_burn=0,resid=np.ones(20),llf=-10.,aic=26.,bic=29.,maroots=np.array([-3.]))
    with patch.object(ARIMA,'fit',return_value=SimpleNamespace(**base)):
        with pytest.raises(ValueError,match='MA residual.*variance'):
            weak_efficiency_test(np.arange(20)**2,np.zeros(20),h=2,lags=5)
    base['params']=np.array([0.,np.nan,1.])
    with patch.object(ARIMA,'fit',return_value=SimpleNamespace(**base)):
        with pytest.raises(ValueError,match='nonfinite parameters'):
            weak_efficiency_test(np.arange(20)**2,np.zeros(20),h=2,lags=5)


def test_result_types_and_structure():
    errors,z,_=regression_sample()
    weak=weak_efficiency_test(errors,np.zeros(100))
    orth=orthogonality_test(errors,np.zeros(100),z)
    assert set(weak)=={'method','null_hypothesis','h','ma_order','lags','statistic','distribution',
                       'pvalue','df','model_df','alpha','reject_null','nobs','diagnostic_nobs',
                       'ma_coefficients','ma_standard_errors','estimation'}
    assert set(orth)=={'method','null_hypothesis','h','nobs','statistic','distribution','pvalue',
                       'alpha','reject_null','cov_type','maxlags','n_restrictions','df_resid',
                       'df','df_num','df_denom','parameters','standard_errors',
                       'confidence_intervals','information_names','bandwidth_covers_overlap'}
    assert isinstance(weak['reject_null'],bool) and isinstance(orth['reject_null'],bool)
    assert weak['h']==orth['h']==1
    assert orth['parameters'].keys()==orth['standard_errors'].keys()==orth['confidence_intervals'].keys()
