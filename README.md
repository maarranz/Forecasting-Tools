# Forecasting-Tools

ATSE forecasting tools.

`static_forecast(results, actual, level=0.95)` evaluates a fitted statsmodels
nonseasonal ARIMA model without user-supplied exogenous regressors. It forecasts
one observation, appends its actual value with `refit=False`, and repeats.

```python
from statsmodels.tsa.arima.model import ARIMA
from forecasting_tools import static_forecast

# y is a numeric Series with a regular date or period index.
results = ARIMA(y.iloc[:80], order=(1, 1, 1)).fit()
evaluation = static_forecast(results, y.iloc[80:], level=0.95)
```

The returned DataFrame preserves evaluation dates and contains `Actual`,
`Forecast`, `Error` (Actual minus Forecast), `SE`, `Lower`, and `Upper`.
Intervals are conditional Gaussian prediction intervals; they exclude parameter
estimation uncertainty. Fitted parameters and the original results stay unchanged.
Built-in constant and time trends are supported.

Evaluation Series indices must be unique, increasing, and continue the model's
recognized training index without gaps. Missing or infinite actuals are rejected.
Array inputs inherit the model's future index. For models fitted to arrays, an
indexed evaluation Series supplies labels for consecutive observation steps.
Irregular training dates ignored by statsmodels are rejected explicitly.

## API overview (Stages 1–6)

Forecast errors are defined as actual minus forecast throughout the API.

| Function | Purpose | Returns |
|---|---|---|
| `static_forecast()` | Sequential one-step ARIMA forecasts with fixed parameters and conditional Gaussian intervals | DataFrame |
| `me()` | Signed mean forecast error (bias) | Float |
| `mae()` | Mean absolute forecast error | Float |
| `mse()` | Mean squared forecast error | Float |
| `rmse()` | Root mean squared forecast error | Float |
| `mape()` | Mean absolute percentage error; rejects zero actuals | Float (percentage) |
| `forecast_accuracy()` | ME, MAE, MSE, RMSE, and MAPE together, in that order | Series |
| `unbiasedness_test()` | Mean-error or joint Mincer–Zarnowitz test; HAC or conventional inference | Dictionary |
| `weak_efficiency_test()` | Ljung–Box diagnostic on errors, or MA(h−1) residuals for multistep forecasts | Dictionary |
| `orthogonality_test()` | Joint test of zero intercept and coefficients on origin-available information; HAC or conventional inference | Dictionary |
| `mgn_test()` | Morgan–Granger–Newbold covariance restriction for two forecasts; equal-MSFE interpretation requires zero population mean errors | Dictionary |
| `dm_test()` | Direct HAC test of equal expected squared or absolute losses, with optional HLN correction | Dictionary |

## Installation

The standard environment uses Python 3.13, pandas below version 3, and
StatsForecast 2.1 or newer, alongside the existing scientific Python and
Jupyter dependencies. From the repository root, create and activate it:

```sh
mamba env create -f environment.yml
mamba activate forecasting_tools
```

If `forecasting_tools` already exists, creating it again will fail. To apply the
specification to an existing environment, first back up its specification and
review the proposed changes:

```sh
mamba env export -n forecasting_tools > forecasting_tools-environment-backup.yml
mamba env update -n forecasting_tools -f environment.yml --dry-run
```

After reviewing the plan, apply it with
`mamba env update -n forecasting_tools -f environment.yml`.
An existing Python 3.14 environment will move to Python 3.13, and pandas 3 will
move to a compatible version below 3. Editing `environment.yml` alone does not
change any installed environment.

For the demonstration notebook, register and select the environment's kernel:

```sh
mamba run -n forecasting_tools python -m ipykernel install --user --name forecasting_tools --display-name "Python (forecasting_tools)"
mamba run -n forecasting_tools jupyter lab Notebooks/Forecasting_Tools_Demo.ipynb
```

Run the tests in the environment:

```sh
mamba run -n forecasting_tools python -m pytest tests -v
```

Tests compare forecasts and standard errors against a single fixed-parameter
filter of the full history for ARIMA(1,0,1) and ARIMA(1,1,1), including date,
period, integer, Series, DataFrame, and array inputs. They also check drift,
intervals, no future-data leakage, unchanged input results, and validation.

## Forecast accuracy evaluation

Use `forecast_accuracy(actual, forecast)` to calculate ME, MAE, MSE, RMSE,
and MAPE together, or use `me`, `mae`, `mse`, `rmse`, and `mape` individually.
Errors are always actual minus forecast. ME measures signed bias: positive
values indicate underforecasting and negative values indicate overforecasting.
Opposite errors can cancel in ME. MAE, MSE, and RMSE measure error magnitude;
smaller values indicate smaller errors. MAE and RMSE use observation units,
while MSE uses squared units and gives large errors extra weight.

This example runs independently:

```python
import pandas as pd
from forecasting_tools import forecast_accuracy

index = pd.date_range('2026-01-01', periods=3, freq='D')
actual = pd.Series([10, 20, 30], index=index)
forecast = pd.Series([8, 23, 26], index=index)
print(forecast_accuracy(actual, forecast).round(3))
# ME        1.000
# MAE       3.000
# MSE       9.667
# RMSE      3.109
# MAPE     16.111
```

For an existing static forecast evaluation, use
`forecast_accuracy(evaluation['Actual'], evaluation['Forecast'])`.

Inputs may be arrays, Python sequences, or Series, and must be one-dimensional,
nonempty, numeric, finite, and equal in length. Two Series must have matching
indexes in the same order; other input pairs are compared positionally.
Observations are never silently realigned or dropped. MAPE is a percentage,
is undefined when any actual value is zero, and can behave poorly near zero.
The combined function also rejects zero actuals because it includes MAPE;
the other four individual functions accept zeros. The same test command above
runs both the accuracy and static forecasting tests.

## Demonstration notebook

[Forecasting-Tools demonstration](Notebooks/Forecasting_Tools_Demo.ipynb) uses
reproducible simulated ARMA data to demonstrate the API through **Stage 5**:
static ARIMA forecasting, forecast accuracy, unbiasedness, weak efficiency,
orthogonality, and Morgan–Granger–Newbold comparison with conventional and HAC
inference. Notebook **v0.3.0** includes tables and five figures. It
integrates the two packages through ARIMA-Tools diagnostics and its documented
Statsmodels estimation workflow; ARIMA-Tools currently has no public fixed-order
fitting function. Forecasting-Tools itself remains independent of ARIMA-Tools.

ARIMA-Tools must be available to execute this particular notebook. Clone it as a
sibling repository. StatsForecast is included in the standard environment to
satisfy ARIMA-Tools' unconditional import; the Forecasting-Tools API itself does
not import or depend internally on StatsForecast. No separate demonstration
environment is needed with the updated specification. Select
`Python (forecasting_tools)` when using the standard environment above.
The notebook supports launching from either the repository root or `Notebooks/`.

## Forecast unbiasedness tests

`unbiasedness_test` provides two regression-based tests with errors defined as
actual minus forecast. The default mean-error test estimates an intercept-only
error regression and tests a zero mean error. Positive estimates indicate
underforecasting; negative estimates indicate overforecasting.

```python
import forecasting_tools as ft

actual = [11, 19, 32, 38]
forecast = [10, 20, 30, 40]
bias = ft.unbiasedness_test(actual, forecast, method="mean_error")
calibration = ft.unbiasedness_test(actual, forecast, method="mincer_zarnowitz")
print(bias["parameters"], bias["pvalue"], bias["reject_null"])
print(calibration["statistic"], calibration["distribution"], calibration["pvalue"])
```

Mincer-Zarnowitz estimates actual = intercept + slope * forecast + disturbance
and **jointly** tests intercept = 0 and slope = 1. Two separate coefficient
checks cannot replace this joint hypothesis.

The default `cov_type="HAC"` uses Newey-West/Bartlett covariance with no
small-sample multiplier. Automatic bandwidth is
`min(n-1, floor(4*(n/100)**(2/9)))`; set `maxlags` to an integer from 0 through
n-1 to choose it explicitly. Observation order represents consecutive time
steps. HAC uses normal mean-error inference and chi-square joint inference
with two restrictions. `cov_type="nonrobust"` uses conventional Student t
inference and a classical joint F-test; leave `maxlags=None` in that case.
`alpha` controls confidence coverage (1-alpha) and the rejection threshold.

Results are dictionaries containing specification, statistic, distribution,
p-value, rejection decision, sample size, bandwidth, parameter estimates,
standard errors, marginal confidence intervals, and degrees of freedom.
Use `parameters["intercept"]` for estimated mean error. Mincer-Zarnowitz also
returns `parameters["slope"]`. See the function docstring for stable result keys.

Inputs follow the accuracy functions' validation rules. Mean-error inference
needs at least two observations; joint inference needs at least three and
nonconstant, numerically identifiable forecasts. Zero residual variance and
singular covariance are rejected with an explanation rather than reporting
misleading p-values. HAC inference is asymptotic and may be unreliable in small
samples; classical exact inference requires independent homoskedastic Gaussian
regression disturbances. Neither covariance choice repairs endogeneity or
first-stage estimation effects. Failure to reject is not proof of unbiasedness,
and unbiasedness is necessary but insufficient for squared-error optimality.

The environment includes pytest. Run all existing and new tests from the
repository root with `mamba run -n forecasting_tools python -m pytest tests -v`.
The earlier unittest tests are discovered by pytest as well.

## Forecast efficiency diagnostics

Weak efficiency checks serial-correlation restrictions. Orthogonality asks
whether specified information available at the forecast origin predicts errors.
Both functions use errors defined as actual minus forecast and return structured
dictionaries with the statistic, reference distribution, p-value, degrees of
freedom, and rejection decision.

```python
import numpy as np
import pandas as pd
import forecasting_tools as ft

rng = np.random.default_rng(984)
origin_signal = pd.Series(rng.normal(size=120), name="origin_signal")
forecast = np.full(120, 100.0)
actual = forecast + 0.4 * origin_signal.to_numpy() + rng.normal(size=120)

serial = ft.weak_efficiency_test(actual, forecast, h=1, lags=10)
information_test = ft.orthogonality_test(
    actual, forecast, origin_signal, h=1, cov_type="HAC"
)
print(serial["statistic"], serial["pvalue"], serial["reject_null"])
print(information_test["parameters"], information_test["pvalue"])

# Overlapping two-step errors can have MA(1) dependence.
innovations = rng.normal(size=121)
two_step_errors = innovations[1:] + 0.4 * innovations[:-1]
overlap = ft.weak_efficiency_test(
    forecast + two_step_errors, forecast, h=2, lags=10
)
print(overlap["ma_coefficients"], overlap["df"], overlap["estimation"])
```

For `h=1`, Ljung–Box is applied directly to errors. For `h>1`, a constant-mean
MA(h-1) is fitted with Statsmodels and Ljung–Box is applied to its filtering
residuals. The degrees-of-freedom adjustment subtracts h-1 estimated MA
coefficients from the diagnostic lag: `df = lags - (h-1)`. The mean and variance
are not additionally deducted under the usual ARMA diagnostic convention.
Diagnostic lags must exceed h-1 and be below the residual sample size; invalid
lags raise an error. Constant errors, failed/nonconverged MA fits, and degenerate
residuals are explained explicitly. Multistep results include MA coefficients,
standard errors, mean, variance, convergence, likelihood criteria, invertibility,
and captured estimation warnings.

This checks compatibility with an MA(h-1) representation, not optimality or the
correctness of the entire covariance structure. Ljung–Box centers its input, and
the multistep model estimates a mean; use the separate unbiasedness test to
examine bias. Small samples, heteroskedasticity, estimated forecasts, and MA
parameters near an invertibility boundary can affect the diagnostic's
approximate reference distribution.

`orthogonality_test` estimates error = intercept + information coefficients +
disturbance, jointly testing the intercept and **all** information coefficients
against zero. Named Series and DataFrame columns retain their names. Arrays
support one variable or an observations-by-variables matrix. The function adds
the intercept, rejects redundant/constant regressors, and requires more
observations than coefficients. With `cov_type="nonrobust"` it uses classical
F inference and t confidence intervals; HAC uses a joint chi-square Wald test
and normal intervals. HAC uses Bartlett weights without a small-sample
multiplier and the Stage 3 bandwidth rule
`min(n-1, floor(4*(n/100)**(2/9)))` unless `maxlags` is specified.

**Avoid look-ahead bias:** each information row must contain only values known
at its forecast origin. A target-date index does not establish availability.
Inputs are neither shifted nor silently aligned, and pandas indexes must match
in the same order. Prepare origin-available values yourself before calling the
function. The horizon is metadata and does not shift observations. For overlapping
h-step forecasts, consider HAC bandwidth at least h-1, and possibly higher if
additional serial dependence is present. The automatic rule does not enforce
that minimum; `bandwidth_covers_overlap` reports whether the chosen bandwidth
covers h-1 lags.

Failure to reject either procedure does not establish forecast optimality:
serial restrictions need not imply independence, supplied information may omit
predictors, nonlinear predictability may remain, and tests can have limited
power. HAC inference does not repair look-ahead bias, endogeneity, or first-stage
estimation effects. See the NumPy-style function docstrings for assumptions and
stable return keys. Run the complete Stage 1–5 suite with the pytest command above.

## Morgan–Granger–Newbold forecast comparison

`mgn_test(actual, forecast1, forecast2)` forms errors e1 = actual − forecast1
and e2 = actual − forecast2, then estimates
`u = intercept + slope * v + disturbance`, where u = e1 + e2 and v = e1 − e2.
It tests **only** the two-sided slope null H0: slope = 0; the intercept remains
unrestricted.

```python
import forecasting_tools as ft

actual = [10, 20, 30, 40, 50]
forecast1 = [9, 22, 27, 39, 48]
forecast2 = [12, 19, 31, 37, 51]
comparison = ft.mgn_test(actual, forecast1, forecast2)  # default HAC inference
print(comparison["parameters"], comparison["statistic"], comparison["pvalue"])
classical = ft.mgn_test(actual, forecast1, forecast2, cov_type="nonrobust")
print(classical["correlation"], classical["statistic"], classical["df"])
```

The default HAC covariance uses Bartlett weights without an additional
small-sample multiplier and bandwidth `min(N-1, floor(4*(N/100)**(2/9)))`,
unless `maxlags` is specified. Slope inference uses an asymptotic normal z
statistic. Conventional OLS inference uses Student t(N−2) and reproduces the
classical correlation statistic `r * sqrt((N-2)/(1-r**2))` for r = Corr(u,v).
Leave `maxlags=None` with conventional covariance. Both choices estimate the
same slope but use different covariance estimators; HAC standard errors need
not be larger. Conventional exact inference requires independent observations
and suitable normal/homoskedastic regression assumptions. HAC permits
heteroskedasticity and weak serial dependence under regularity conditions.
HAC inference is asymptotic;
the short example illustrates syntax, not reliable small-sample evidence.

**MSFE interpretation requires both errors to have zero population means.**
With this assumption, the slope null corresponds to equality of population
mean squared forecast errors. Without it, the regression tests equality of
centered error variances—a covariance restriction—not equality of expected
squared losses. Failure to reject an unbiasedness test does not establish the
zero-mean assumption. Direct loss-differential testing is provided by `dm_test()` in Stage 6.

The result dictionary reports coefficients, standard errors, marginal confidence
intervals, correlation, matched statistic/distribution, p-value, degrees of
freedom, rejection decision, and selected bandwidth. `mean_errors`, `mse`, and
`mse_difference` provide descriptive summaries; the difference is model1 minus
model2, so positive values indicate smaller sample MSE for model2. These summaries
are not used to construct an additional test. The result also carries an explicit
`msfe_interpretation_note`.

All observations must refer to the same evaluation periods. Any two pandas
Series among the three inputs must have matching indexes in the same order;
other inputs are positional. Nothing is silently truncated, reordered, or
removed. At least three observations are needed. Constant error sums/differences,
perfect or near-perfect collinearity, zero residual variance, and invalid
covariance are rejected with informative errors. Failure to reject MGN does not
prove equal forecasting performance or optimality. The full Stage 1–5 pytest
suite uses the testing command above; no new dependencies are required.

## Diebold–Mariano forecast comparison (Stage 6)

```python
dm_test(actual, forecast1, forecast2, loss="squared", h=1,
        alternative="two-sided", maxlags=None, hln=False, alpha=0.05)
```

Errors are actual minus forecast. The loss differential is
`d = L(error1) - L(error2)`, where `loss="squared"` uses squared errors and
`loss="absolute"` uses absolute errors. The null is **E(d) = 0**.
Negative observed differentials favor model 1; positive values favor model 2.
Statistical significance is determined by the test, not by the sign alone.
`alternative="two-sided"` tests unequal expected losses; `"less"` tests lower
expected loss for model 1, and `"greater"` lower expected loss for model 2.
Swapping forecasts reverses the statistic and exchanges the one-sided meanings.

The statistic is calculated **directly** as `DM = mean(d) / SE_HAC(mean(d))`.
For centered d, lag-k autocovariances use divisor N, and the long-run variance is
`Omega = gamma_0 + 2*sum((1-k/(q+1))*gamma_k, k=1,...,q)`.
The standard error is `sqrt(Omega/N)`. Bartlett weights and no small-sample
covariance multiplier match the existing API conventions. This equals the HAC
statistic for the intercept in `d = intercept + disturbance` with identical
bandwidth and normalization; an independent Statsmodels regression validates
the calculation in tests, but is not the public function's computational path.

`maxlags=None` selects `min(N-1, floor(4*(N/100)**(2/9)))`; explicit integer
bandwidths must be between 0 and N-1. The positive integer horizon `h` does not
shift data or override bandwidths. For overlapping h-step forecasts, h-1 is a
conventional bandwidth under suitable assumptions; additional dependence can
require more lags even for h=1. The automatic rule is a practical default, not
a universally optimal choice.

By default, `hln=False` reports conventional DM with standard normal inference.
With `hln=True`, the Harvey–Leybourne–Newbold correction multiplies DM by
`sqrt((N+1-2*h+h*(h-1)/N)/N)` and uses Student t(N-1) inference. HLN requires
h < N and a finite, strictly positive factor. It leaves the losses, long-run
variance, and standard error unchanged. This finite-sample adjustment was
developed under particular forecasting assumptions; it does not guarantee exact
inference under arbitrary dependence or heteroskedasticity.

This example runs independently:

```python
import forecasting_tools as ft

actual = [10, 20, 30, 40, 50]
forecast1 = [9, 22, 27, 39, 48]
forecast2 = [12, 19, 31, 37, 51]
result = ft.dm_test(actual, forecast1, forecast2, loss="squared", h=1,
                    alternative="two-sided", hln=False)
result_hln = ft.dm_test(actual, forecast1, forecast2, loss="squared", h=1,
                        hln=True)
print(result["dm_statistic"], result["pvalue"])
print(result_hln["statistic"], result_hln["pvalue"])
absolute = ft.dm_test(actual, forecast1, forecast2, loss="absolute")
```

Results include `n`, `h`, `loss`, `alternative`, `mean_loss1`, `mean_loss2`,
`mean_loss_diff`, `long_run_variance`, `se`, `dm_statistic`, `statistic`,
`pvalue`, `maxlags`, `hln`, `hln_factor`, `df`, `distribution`, `alpha`, and
`reject_null`. `dm_statistic` always retains conventional DM; `statistic` is
identical unless HLN is enabled. `alpha` controls the rejection threshold.

Inputs must be finite, numeric, equal-length one-dimensional samples with at
least two observations. Any pair of Series must have matching indexes in the
same order; other inputs pair positionally. Missing observations are never
removed or realigned. Zero, nonpositive, nonfinite or numerically degenerate
long-run variance and overflowing calculations raise informative errors.

Unlike MGN's covariance restriction, **DM directly tests equality of expected
loss without a zero-mean forecast-error assumption**. HAC inference still
requires suitable stationarity, moments and weak dependence; small samples,
bandwidth selection and first-stage estimation effects require care.
Nonrejection does not prove equal accuracy or optimality. Run all Stage 1–6
tests with the existing pytest command. The demonstration notebook remains
v0.3.0 and covers Stages 1–5; Stage 6 is not yet added to it.
