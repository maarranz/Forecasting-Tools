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
