"""Forecast evaluation helpers for statsmodels ARIMA models."""

import numbers

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults, ARIMAResultsWrapper


def static_forecast(results, actual, level=0.95):
    """Evaluate sequential one-step forecasts with fixed fitted parameters.

    Parameters
    ----------
    results : statsmodels ARIMAResults or ARIMAResultsWrapper
        Fitted, nonseasonal, univariate ARIMA without user-supplied exog.
        Built-in constant and time trends are supported.
    actual : pandas.Series or one-dimensional array-like
        Finite evaluation observations in chronological order. A Series must
        have a unique, increasing index. For a model with a supported index,
        its index must continue the training index without gaps. Arrays inherit
        the model's forecast index (integer positions for undated models).
        A date index ignored by statsmodels because it lacks a frequency is
        rejected: fit with a regular date/period index or with array data.
        For array-fitted models, a Series may supply evaluation date labels;
        these are labels for consecutive observation steps.
    level : float, default 0.95
        Coverage strictly between zero and one.

    Returns
    -------
    pandas.DataFrame
        Actual, Forecast, Error (Actual - Forecast), SE, Lower, Upper.
        Intervals are conditional Gaussian prediction intervals, excluding
        parameter-estimation uncertainty. Each actual is appended only after
        its prediction, with refit=False. The supplied results are unchanged.
        Empty evaluation data returns an empty frame with the same columns.

    Notes
    -----
    statsmodels append refilters the growing history, so this implementation
    favors transparent fixed-parameter behavior over speed on long datasets.
    """
    if not isinstance(results, (ARIMAResults, ARIMAResultsWrapper)) or not isinstance(results.model, ARIMA):
        raise TypeError("results must be fitted statsmodels ARIMA results")
    model = results.model
    if any(model.seasonal_order[:3]):
        raise ValueError("seasonal ARIMA models are not supported")
    # ARIMA's exog includes its built-in trend; only reject user regressors.
    if model._spec_arima.k_exog:
        raise ValueError("models with user-supplied exog are not supported")
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, numbers.Real) or not 0 < level < 1:
        raise ValueError("level must be a finite number strictly between 0 and 1")
    if results.filter_results is None or results.nobs < 1:
        raise ValueError("results must retain fitted filtering data")

    if isinstance(actual, pd.DataFrame):
        raise TypeError("actual must be a Series or one-dimensional array-like")
    raw = np.asarray(actual)
    if raw.ndim != 1 or np.iscomplexobj(raw) or raw.dtype.kind == 'b':
        raise ValueError("actual must contain one-dimensional real numeric observations")
    try:
        values = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual must contain real numeric observations") from exc
    if not np.isfinite(values).all():
        raise ValueError("actual must contain finite observations, without missing values")

    original_index = model.data.row_labels
    if original_index is not None and (not original_index.is_unique or not original_index.is_monotonic_increasing):
        raise ValueError("training index must be unique and increasing")
    if isinstance(original_index, (pd.DatetimeIndex, pd.PeriodIndex)) and model._index_generated:
        raise ValueError("training date index must have a regular frequency recognized by statsmodels")

    count = len(values)
    expected = None
    if count:
        _, _, _, expected = model._get_prediction_index(results.nobs, results.nobs + count - 1)
        if expected is None:
            expected = pd.RangeIndex(results.nobs, results.nobs + count)
    else:
        expected = model._index[:0]
    index = actual.index.copy() if isinstance(actual, pd.Series) else expected
    if not index.is_unique or not index.is_monotonic_increasing or index.hasnans:
        raise ValueError("evaluation index must be unique, increasing, and nonmissing")
    if count and original_index is not None and not model._index_generated and not index.equals(expected):
        raise ValueError("evaluation index must immediately continue the training index without gaps")

    rows = []
    current = results
    z = norm.isf((1 - float(level)) / 2)
    for value in values:
        prediction = current.get_forecast(steps=1)
        forecast = float(np.asarray(prediction.predicted_mean).reshape(-1)[0])
        se = float(np.asarray(prediction.se_mean).reshape(-1)[0])
        rows.append((value, forecast, value - forecast, se, forecast - z * se, forecast + z * se))
        # Match the model's own index and endogenous name, not actual.name.
        # append can turn a Series-backed model into a DataFrame-backed model.
        original = current.model.data.orig_endog
        if isinstance(original, (pd.Series, pd.DataFrame)):
            _, _, _, next_index = current.model._get_prediction_index(current.nobs, current.nobs)
            if isinstance(original, pd.DataFrame):
                observation = pd.DataFrame([[value]], index=next_index, columns=original.columns)
            else:
                observation = pd.Series([value], index=next_index, name=original.name)
        else:
            observation = np.array([value])
        current = current.append(observation, refit=False)
    return pd.DataFrame(rows, index=index, columns=["Actual", "Forecast", "Error", "SE", "Lower", "Upper"], dtype=float)
