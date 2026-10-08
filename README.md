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

Run the tests in the existing environment:

```sh
mamba run -n forecasting_tools python -m unittest discover -s tests -v
```

Tests compare forecasts and standard errors against a single fixed-parameter
filter of the full history for ARIMA(1,0,1) and ARIMA(1,1,1), including date,
period, integer, Series, DataFrame, and array inputs. They also check drift,
intervals, no future-data leakage, unchanged input results, and validation.
