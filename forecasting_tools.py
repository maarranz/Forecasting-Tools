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


def _accuracy_inputs(actual, forecast):
    """Validate pairs without dropping observations or aligning pandas labels."""
    if isinstance(actual, pd.Series) and isinstance(forecast, pd.Series):
        if not actual.index.equals(forecast.index):
            raise ValueError("actual and forecast Series indexes must match in the same order")
    arrays = []
    for name, sample in (("actual", actual), ("forecast", forecast)):
        if isinstance(sample, pd.DataFrame):
            raise TypeError(f"{name} must be a one-dimensional array, sequence, or Series")
        if np.ma.isMaskedArray(sample) and np.ma.getmaskarray(sample).any():
            raise ValueError(f"{name} contains masked observations; missing values are not allowed")
        try:
            raw = np.asarray(sample)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional numeric sample") from exc
        if raw.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if raw.size == 0:
            raise ValueError(f"{name} must be nonempty")
        if pd.isna(raw).any():
            raise ValueError(f"{name} contains missing observations; all values must be finite")
        if raw.dtype.kind not in 'iufO':
            raise TypeError(f"{name} must contain real numeric observations, not strings, booleans, or complex values")
        if raw.dtype.kind == 'O' and any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            for value in raw
        ):
            raise TypeError(f"{name} must contain real numeric observations")
        try:
            values = raw.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} values must be representable as finite real numbers") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite observations; all values must be finite")
        arrays.append(values)
    if len(arrays[0]) != len(arrays[1]):
        raise ValueError("actual and forecast must have equal lengths")
    return tuple(arrays)


def _accuracy_mape(actual, forecast):
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when any actual observation is zero")
    return float(100 * np.mean(np.abs((actual - forecast) / actual)))


def me(actual, forecast):
    """Calculate mean error (ME), a measure of signed forecast bias.

    With errors e_i = actual_i - forecast_i, ME = sum(e_i) / n.

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real observations.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real predictions of equal length. If both inputs are Series,
        their indexes must match in the same order; otherwise pairing is
        positional. Missing observations are never removed.

    Returns
    -------
    float
        Mean signed error, in the same units as the actual observations.
        Positive ME indicates underforecasting; negative ME indicates
        overforecasting. Zero indicates no average signed bias.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        If samples are empty, nonfinite, not one-dimensional, unequal in
        length, or have mismatched Series indexes.

    Notes
    -----
    Positive and negative errors can cancel. A small ME does not imply small
    error magnitude or accurate individual forecasts. ME is scale-dependent.

    Examples
    --------
    >>> me([10, 20, 30], [8, 23, 26])
    1.0
    """
    actual, forecast = _accuracy_inputs(actual, forecast)
    return float(np.mean(actual - forecast))


def mae(actual, forecast):
    """Calculate mean absolute error (MAE), a measure of error magnitude.

    With e_i = actual_i - forecast_i, MAE = sum(abs(e_i)) / n.

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real observations.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real predictions of equal length. Two Series must have matching
        indexes in the same order; other pairs are positional. Missing values
        are rejected, never removed.

    Returns
    -------
    float
        Nonnegative average absolute error in the actual observations' units.
        Smaller values indicate smaller errors; zero means perfect forecasts.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        If samples are empty, nonfinite, not one-dimensional, unequal in
        length, or have mismatched Series indexes.

    Notes
    -----
    MAE measures magnitude, not signed bias: opposite signs do not cancel.
    It is scale-dependent and cannot directly compare differently scaled
    series. It weights each unit of absolute error equally.

    Examples
    --------
    >>> mae([10, 20, 30], [8, 23, 26])
    3.0
    """
    actual, forecast = _accuracy_inputs(actual, forecast)
    return float(np.mean(np.abs(actual - forecast)))


def mse(actual, forecast):
    """Calculate mean squared error (MSE), a measure of error magnitude.

    With e_i = actual_i - forecast_i, MSE = sum(e_i ** 2) / n.
    The denominator is n, with no degrees-of-freedom adjustment.

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real observations.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real predictions of equal length. Two Series must have matching
        indexes in the same order; other pairs are positional. Missing values
        are rejected, never removed.

    Returns
    -------
    float
        Nonnegative average squared error in squared observation units.
        Smaller values indicate smaller errors; zero means perfect forecasts.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        If samples are empty, nonfinite, not one-dimensional, unequal in
        length, or have mismatched Series indexes.

    Notes
    -----
    MSE measures magnitude, not signed bias. Squaring gives large errors more
    weight and makes it sensitive to outliers. It is scale-dependent, uses
    squared units, and may overflow floating-point arithmetic for huge errors.

    Examples
    --------
    >>> mse([10, 20, 30], [8, 23, 26])
    9.666666666666666
    """
    actual, forecast = _accuracy_inputs(actual, forecast)
    return float(np.mean(np.square(actual - forecast)))


def rmse(actual, forecast):
    """Calculate root mean squared error (RMSE), an error magnitude measure.

    With e_i = actual_i - forecast_i, RMSE = sqrt(sum(e_i ** 2) / n).

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real observations.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real predictions of equal length. Two Series must have matching
        indexes in the same order; other pairs are positional. Missing values
        are rejected, never removed.

    Returns
    -------
    float
        Nonnegative root mean squared error in the observation units.
        Smaller values indicate smaller errors; zero means perfect forecasts.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        If samples are empty, nonfinite, not one-dimensional, unequal in
        length, or have mismatched Series indexes.

    Notes
    -----
    RMSE measures magnitude, not signed bias. It emphasizes large errors and
    is sensitive to outliers. It is scale-dependent, so comparisons across
    differently scaled series can mislead. Squaring huge errors can overflow.

    Examples
    --------
    >>> round(rmse([10, 20, 30], [8, 23, 26]), 6)
    3.109126
    """
    return float(np.sqrt(mse(actual, forecast)))


def mape(actual, forecast):
    """Calculate mean absolute percentage error (MAPE), in percent.

    With e_i = actual_i - forecast_i,
    MAPE = (100 / n) * sum(abs(e_i / actual_i)).

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real observations, all nonzero.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real predictions of equal length. Two Series must have matching
        indexes in the same order; other pairs are positional. Missing values
        are rejected, never removed.

    Returns
    -------
    float
        Nonnegative mean absolute percentage error. For example, 10.0 means
        10 percent. Smaller values indicate smaller relative errors; zero
        means perfect forecasts. Values can exceed 100 percent.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        If actual contains zero, or samples are empty, nonfinite, not
        one-dimensional, unequal in length, or have mismatched Series indexes.

    Notes
    -----
    MAPE measures relative magnitude, not signed bias. It is undefined for zero
    actuals and can behave poorly near zero, where small absolute errors become
    huge percentages. It is most interpretable for positive ratio-scale data
    with a meaningful zero, and penalizes over/underforecasting asymmetrically.
    Negative actuals are mathematically allowed through the absolute ratio.

    Examples
    --------
    >>> round(mape([10, 20, 30], [8, 23, 26]), 6)
    16.111111
    """
    actual, forecast = _accuracy_inputs(actual, forecast)
    return _accuracy_mape(actual, forecast)


def forecast_accuracy(actual, forecast):
    """Calculate ME, MAE, MSE, RMSE, and percentage MAPE together.

    For n paired observations with e_i = actual_i - forecast_i:
    ME = sum(e_i) / n; MAE = sum(abs(e_i)) / n;
    MSE = sum(e_i ** 2) / n; RMSE = sqrt(MSE);
    MAPE = (100 / n) * sum(abs(e_i / actual_i)).

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real observations, all nonzero
        because the combined result includes MAPE.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real predictions of equal length. Two Series must have matching
        indexes in the same order; other pairs are positional. Missing values
        are rejected, never removed or silently aligned.

    Returns
    -------
    pandas.Series
        Float entries indexed by ME, MAE, MSE, RMSE, MAPE in that order.
        ME measures signed bias: positive means underforecasting, negative
        means overforecasting. MAE, MSE, RMSE, and MAPE measure magnitude,
        with smaller values indicating smaller errors. ME, MAE, and RMSE
        use observation units, MSE uses squared units, and MAPE uses percent.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        If actual contains zero, or samples are empty, nonfinite, not
        one-dimensional, unequal in length, or have mismatched Series indexes.

    Notes
    -----
    Signed errors can cancel in ME, so zero bias need not mean accuracy.
    MAE, MSE, and RMSE are scale-dependent; MSE and RMSE emphasize large errors
    and squaring can overflow for huge values. MAPE is undefined at zero and
    unstable near zero; it is most meaningful for positive ratio-scale data.
    Use individual functions when only a subset of the measures is needed.

    Examples
    --------
    >>> accuracy = forecast_accuracy([10, 20, 30], [8, 23, 26])
    >>> accuracy.index.tolist()
    ['ME', 'MAE', 'MSE', 'RMSE', 'MAPE']
    >>> float(accuracy['MAE'])
    3.0
    """
    actual, forecast = _accuracy_inputs(actual, forecast)
    percentage = _accuracy_mape(actual, forecast)
    errors = actual - forecast
    squared = float(np.mean(np.square(errors)))
    return pd.Series(
        [float(np.mean(errors)), float(np.mean(np.abs(errors))), squared,
         float(np.sqrt(squared)), percentage],
        index=["ME", "MAE", "MSE", "RMSE", "MAPE"], dtype=float,
    )


def unbiasedness_test(actual, forecast, method="mean_error", cov_type="HAC",
                      maxlags=None, alpha=0.05):
    r"""Test forecast unbiasedness using mean errors or joint calibration.

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional, finite real observations in observation
        order. Zeros are allowed. No observations are dropped.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite real forecasts of the same length. Two Series must have identical
        indexes in the same order; other input pairs are matched positionally.
    method : {"mean_error", "mincer_zarnowitz"}, default "mean_error"
        ``mean_error`` estimates e_t = a + u_t, where e_t = y_t - forecast_t,
        and tests H0: a = 0 against H1: a != 0. At least two pairs are required.
        ``mincer_zarnowitz`` estimates y_t = a + b * forecast_t + u_t and tests
        the joint H0: a = 0, b = 1 against the alternative that at least one
        restriction fails. At least three pairs and a full-rank design are
        required. This is a joint test, not two separate coefficient tests.
    cov_type : {"HAC", "nonrobust"}, default "HAC"
        ``HAC`` uses Newey-West/Bartlett covariance without a small-sample
        multiplier. Mean-error inference uses an asymptotic normal z statistic;
        joint calibration uses asymptotic chi-square with two restrictions.
        Coefficient confidence intervals use normal critical values.
        ``nonrobust`` uses conventional OLS covariance, Student t inference
        and intervals with n-k residual degrees of freedom. Joint calibration
        uses the classical F(2, n-2) test (Wald quadratic form divided by 2).
    maxlags : int or None, default None
        HAC bandwidth L. None selects min(n-1, floor(4*(n/100)**(2/9))),
        following Statsmodels' automatic Newey-West convention. Bartlett
        weights are 1-j/(L+1) for j=1,...,L. Explicit L must be a nonnegative
        integer smaller than n; booleans are rejected. For nonrobust covariance,
        only None is accepted because no HAC bandwidth is used.
    alpha : float, default 0.05
        Significance level strictly between zero and one. Confidence intervals
        have coverage 1-alpha; reject_null is True when pvalue < alpha.
        This argument is not the regression intercept a.

    Returns
    -------
    dict
        Stable keys shared by both methods:

        * ``method``, ``null_hypothesis``, ``cov_type``: specification strings.
        * ``statistic``, ``distribution`` ("t", "normal", "F", or "chi2"),
          ``pvalue``: matched test statistic, reference distribution, and
          two-sided mean-error or upper-tail joint-test probability.
        * ``alpha``, ``reject_null``, ``nobs``: float, bool, and int.
        * ``maxlags``: selected HAC bandwidth, or None for conventional OLS.
        * ``parameters``, ``standard_errors``: dictionaries of floats keyed by
          "intercept" and, for Mincer-Zarnowitz, "slope". In the mean-error
          method, "intercept" is the estimated mean error.
        * ``confidence_intervals``: same parameter keys, each mapping to a
          (lower, upper) tuple. These are marginal coefficient intervals and
          do not replace the joint test of (a,b)=(0,1).
        * ``n_restrictions``: 1 or 2; ``df_resid``: n-k.
        * ``df``: t degrees of freedom or chi-square restrictions, otherwise
          None. ``df_num``, ``df_denom``: F degrees of freedom, otherwise None.

    Raises
    ------
    TypeError
        If observations are not real numeric values.
    ValueError
        For invalid options, invalid or insufficient samples, constant or
        numerically rank-deficient calibration designs, or degenerate residual
        variance/covariance for which regression inference is undefined.

    Notes
    -----
    A positive mean-error intercept indicates average underforecasting; a
    negative value indicates overforecasting. Rejection provides evidence
    against the specified null. Failure to reject is not proof of unbiasedness:
    small samples or imprecise estimates may have little power. Unconditional
    zero mean errors alone do not establish calibration. Mincer-Zarnowitz tests
    the stronger joint intercept/slope restriction, not full conditional
    efficiency. Unbiasedness is necessary but insufficient for forecast
    optimality under squared-error loss and the stated information set.

    Classical t/F inference assumes a correctly specified regression with
    independent, homoskedastic Gaussian disturbances for exact finite-sample
    reference distributions. HAC allows heteroskedasticity and weak serial
    dependence but remains asymptotic, requires suitable moment/dependence
    conditions, and can be unreliable in small samples. Observation order must
    represent consecutive time steps; index labels are not used to detect gaps.
    HAC does not repair endogeneity or account automatically for first-stage
    parameter estimation. Estimated forecasts, nested models, and unusual
    designs may require more specialized inference.

    Examples
    --------
    >>> result = unbiasedness_test([11, 19, 32, 38], [10, 20, 30, 40],
    ...                           cov_type="nonrobust")
    >>> round(result['parameters']['intercept'], 6)
    0.0
    >>> result['reject_null']
    False
    >>> result = unbiasedness_test([11, 19, 32, 38], [10, 20, 30, 40],
    ...                           method="mincer_zarnowitz", maxlags=1)
    >>> result['distribution'], result['n_restrictions']
    ('chi2', 2)
    """
    from statsmodels.regression.linear_model import OLS

    if not isinstance(method, str) or method not in ("mean_error", "mincer_zarnowitz"):
        raise ValueError("method must be 'mean_error' or 'mincer_zarnowitz'")
    if not isinstance(cov_type, str) or cov_type not in ("HAC", "nonrobust"):
        raise ValueError("cov_type must be 'HAC' or 'nonrobust'")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real)
            or not 0 < alpha < 1):
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if maxlags is not None:
        if (isinstance(maxlags, (bool, np.bool_))
                or not isinstance(maxlags, numbers.Integral) or maxlags < 0):
            raise ValueError("maxlags must be None or a nonnegative integer")
        if cov_type == "nonrobust":
            raise ValueError("maxlags is only applicable to HAC; use None with nonrobust")

    actual, forecast = _accuracy_inputs(actual, forecast)
    n = len(actual)
    joint = method == "mincer_zarnowitz"
    k = 2 if joint else 1
    if n <= k:
        raise ValueError(f"{method} requires at least {k + 1} observations")
    if maxlags is not None and maxlags >= n:
        raise ValueError("HAC maxlags must be smaller than the number of observations")
    x = np.column_stack((np.ones(n), forecast)) if joint else np.ones((n, 1))
    if np.linalg.matrix_rank(x) != k:
        raise ValueError("Mincer-Zarnowitz design is rank-deficient; forecasts must vary and be numerically identifiable")
    with np.errstate(over="raise", invalid="raise"):
        try:
            dependent = actual if joint else actual - forecast
        except FloatingPointError as exc:
            raise ValueError("forecast errors overflow; rescale observations before inference") from exc
    fit = OLS(dependent, x, missing="raise").fit(use_t=True)
    # Exact fits can leave rounding-sized residuals and spurious finite tests.
    if np.linalg.norm(fit.resid) <= n * np.finfo(float).eps * max(1.0, np.linalg.norm(dependent)):
        raise ValueError("Residual variance is zero or numerically degenerate; unbiasedness inference is undefined")
    lags = None
    if cov_type == "HAC":
        lags = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9)))) if maxlags is None else int(maxlags)
        fit = fit.get_robustcov_results(
            cov_type="HAC", use_t=False, maxlags=lags,
            kernel="bartlett", use_correction=False,
        )
    covariance = np.asarray(fit.cov_params())
    if not np.isfinite(covariance).all() or np.linalg.matrix_rank(covariance) != k:
        raise ValueError("Parameter covariance is nonfinite or singular; inference is undefined")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Parameter covariance is not positive definite; inference is undefined") from exc
    if joint:
        test = fit.wald_test((np.eye(2), np.array([0.0, 1.0])),
                             use_f=cov_type == "nonrobust", scalar=True)
        statistic = float(test.statistic)
        pvalue = float(test.pvalue)
        distribution = "F" if cov_type == "nonrobust" else "chi2"
    else:
        test = fit.t_test(np.ones((1, 1)), use_t=cov_type == "nonrobust")
        statistic = float(np.asarray(test.statistic).item())
        pvalue = float(np.asarray(test.pvalue).item())
        distribution = "t" if cov_type == "nonrobust" else "normal"
    intervals = np.asarray(fit.conf_int(alpha=float(alpha)))
    if not np.isfinite([statistic, pvalue]).all() or not np.isfinite(intervals).all():
        raise ValueError("Nonfinite inference results; rescale observations or revise the design")
    names = ["intercept", "slope"] if joint else ["intercept"]
    df_resid = n - k
    return {
        "method": method,
        "null_hypothesis": "intercept = 0, slope = 1" if joint else "mean_error = 0",
        "statistic": statistic, "distribution": distribution, "pvalue": pvalue,
        "alpha": float(alpha), "reject_null": bool(pvalue < alpha), "nobs": n,
        "cov_type": cov_type, "maxlags": lags,
        "parameters": dict(zip(names, map(float, fit.params))),
        "standard_errors": dict(zip(names, map(float, fit.bse))),
        "confidence_intervals": dict(zip(names, [tuple(map(float, row)) for row in intervals])),
        "n_restrictions": k, "df_resid": df_resid,
        "df": df_resid if distribution == "t" else k if distribution == "chi2" else None,
        "df_num": k if distribution == "F" else None,
        "df_denom": df_resid if distribution == "F" else None,
    }


def _efficiency_options(h, alpha):
    """Validate horizon and significance level for efficiency diagnostics."""
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, numbers.Integral) or h < 1:
        raise ValueError("h must be a positive integer forecast horizon")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real)
            or not 0 < alpha < 1):
        raise ValueError("alpha must be a finite number strictly between zero and one")


def _efficiency_errors(actual, forecast):
    """Validate observations and compute errors without numerical overflow."""
    actual_values, forecast_values = _accuracy_inputs(actual, forecast)
    with np.errstate(over="raise", invalid="raise"):
        try:
            return actual_values - forecast_values
        except FloatingPointError as exc:
            raise ValueError("Forecast errors overflow; rescale observations") from exc


def _efficiency_variation(values, label):
    """Reject absent or numerically negligible residual variation."""
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite values")
    centered = values - values.mean()
    if (not np.isfinite(centered).all()
            or np.linalg.norm(centered) <= len(values) * np.finfo(float).eps
            * max(1.0, np.linalg.norm(values))):
        raise ValueError(f"{label} variance is zero or numerically degenerate")


def weak_efficiency_test(actual, forecast, h=1, lags=10, alpha=0.05):
    r"""Check serial-correlation restrictions on consecutive forecast errors.

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        One-dimensional finite real evaluation observations, in time order.
    forecast : numpy.ndarray, sequence, or pandas.Series
        Finite forecasts of equal length, with errors e_(t+h|t) = actual -
        forecast. Two Series must have identical indexes in the same order;
        other pairs are positional. Missing observations are never removed.
    h : int, default 1
        Positive forecast horizon. h=1 applies Ljung-Box directly to errors.
        h>1 estimates ARIMA(0,0,h-1) with a constant mean and invertibility
        enforced, then applies Ljung-Box to its one-step filtering residuals.
        Errors must correspond to forecasts issued at consecutive origins.
    lags : int, default 10
        Single positive diagnostic lag m: autocorrelations at lags 1,...,m
        enter one portmanteau test, rather than m separate tests. Must be
        smaller than the diagnostic sample size and greater than h-1.
        Invalid lags are rejected, not silently omitted. Multistep estimation
        also requires n > (h-1)+2, leaving observations beyond the estimated
        MA coefficients, mean, and variance parameters.
    alpha : float, default 0.05
        Significance level strictly between zero and one. Reject when
        pvalue < alpha.

    Returns
    -------
    dict
        ``method`` ("weak_efficiency"), ``null_hypothesis``, ``h``,
        ``ma_order`` (h-1), ``lags``, ``statistic``, ``distribution`` ("chi2"),
        ``pvalue``, ``df`` (lags-ma_order), ``model_df`` (ma_order), ``alpha``,
        ``reject_null``, ``nobs`` (input size), and ``diagnostic_nobs``.
        ``ma_coefficients`` maps MA parameter names (ma.L1, etc.) to floats;
        ``ma_standard_errors`` contains their estimated standard errors.
        These dictionaries are empty for h=1. ``estimation`` is None for h=1;
        otherwise it contains ``converged``, ``mean``, ``sigma2``, ``llf``,
        ``aic``, ``bic``, ``iterations`` (int or None), ``invertible``,
        ``residual_burn`` and a list of captured ``warnings``. No fitted
        Statsmodels object is exposed through this return structure.

    Raises
    ------
    TypeError
        For nonnumeric observations.
    ValueError
        For invalid inputs/options, insufficient data/lags, zero residual
        variance, failed or nonconverged MA estimation, or nonfinite inference.

    Notes
    -----
    Ljung-Box uses centered diagnostic residuals and
    Q = n(n+2) * sum(r_j**2 / (n-j), j=1,...,m), with asymptotic chi-square
    reference. After estimating MA(q), ``model_df=q`` gives m-q degrees of
    freedom; the fitted mean and innovation variance are not additionally
    subtracted, following the usual ARMA residual-diagnostic convention.
    Residuals exclude the fitted model's loglikelihood_burn observations; for
    this stationary MA specification that value is normally zero.

    Under suitable squared-error optimality and information-set assumptions,
    one-step errors are uncorrelated and overlapping h-step errors may have
    dependence through lag h-1. This procedure checks compatibility with an
    MA(h-1) representation and residual serial-correlation restrictions. It
    does not establish optimality or the entire covariance structure. It does
    not test zero bias: a fitted constant absorbs the multistep error mean and
    Ljung-Box demeaning similarly removes the one-step mean. Use a separate
    unbiasedness test. Uncorrelatedness also does not imply independence.

    Reference probabilities are approximate, especially in small samples,
    with conditional heteroskedasticity, near-boundary MA parameters, or when
    forecast parameter estimation matters. Failure to reject may reflect low
    power. Horizon metadata and information timing cannot be inferred from
    input values; no dates are shifted or automatically aligned.

    Examples
    --------
    >>> rng = np.random.default_rng(41)
    >>> errors = rng.normal(size=100)
    >>> result = weak_efficiency_test(errors, np.zeros(100), lags=5)
    >>> result['df'], result['ma_order']
    (5, 0)
    >>> innovations = rng.normal(size=121)
    >>> errors = innovations[1:] + 0.4 * innovations[:-1]
    >>> result = weak_efficiency_test(errors, np.zeros(120), h=2, lags=8)
    >>> result['df'], result['estimation']['converged']
    (7, True)
    """
    import warnings
    from statsmodels.stats.diagnostic import acorr_ljungbox

    _efficiency_options(h, alpha)
    if isinstance(lags, (bool, np.bool_)) or not isinstance(lags, numbers.Integral) or lags < 1:
        raise ValueError("lags must be a positive integer diagnostic lag")
    q = int(h) - 1
    if lags <= q:
        raise ValueError("Diagnostic lags must exceed the fitted MA order h-1; Ljung-Box needs positive degrees of freedom")
    errors = _efficiency_errors(actual, forecast)
    n = len(errors)
    if lags >= n:
        raise ValueError("Diagnostic lags must be smaller than the number of observations")
    if q and n <= q + 2:
        raise ValueError("MA estimation requires more than ma_order+2 observations")
    _efficiency_variation(errors, "Forecast error")
    diagnostic = errors
    coefficients, standard_errors, estimation = {}, {}, None
    if q:
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fit = ARIMA(errors, order=(0, 0, q), trend="c",
                            enforce_invertibility=True).fit()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as exc:
            raise ValueError(f"MA({q}) estimation failed: {exc}") from exc
        if not fit.mle_retvals.get("converged", False):
            raise ValueError(f"MA({q}) estimation did not converge; revise the specification or sample")
        if not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all():
            raise ValueError("MA estimation returned nonfinite parameters or standard errors")
        parameters = dict(zip(fit.param_names, map(float, fit.params)))
        if parameters.get("sigma2", 0) <= 0:
            raise ValueError("Estimated MA innovation variance must be positive")
        burn = int(fit.loglikelihood_burn)
        diagnostic = np.asarray(fit.resid, dtype=float)[burn:]
        coefficients = dict(zip([f"ma.L{j}" for j in range(1, q + 1)], map(float, fit.maparams)))
        errors_by_name = dict(zip(fit.param_names, map(float, fit.bse)))
        standard_errors = {name: errors_by_name[name] for name in coefficients}
        estimation = {
            "converged": True, "mean": parameters["const"], "sigma2": parameters["sigma2"],
            "llf": float(fit.llf), "aic": float(fit.aic), "bic": float(fit.bic),
            "iterations": int(fit.mle_retvals["iterations"]) if "iterations" in fit.mle_retvals else None,
            "invertible": bool(np.all(np.abs(fit.maroots) > 1)),
            "residual_burn": burn, "warnings": [str(w.message) for w in captured],
        }
        if not np.isfinite([fit.llf, fit.aic, fit.bic]).all():
            raise ValueError("MA estimation returned nonfinite likelihood diagnostics")
        if lags >= len(diagnostic):
            raise ValueError("Diagnostic lags must be smaller than the post-initialization residual sample")
        _efficiency_variation(diagnostic, "MA residual")
    test = acorr_ljungbox(diagnostic, lags=[int(lags)], model_df=q, return_df=True)
    statistic, pvalue = map(float, test.iloc[0][["lb_stat", "lb_pvalue"]])
    if not np.isfinite([statistic, pvalue]).all():
        raise ValueError("Ljung-Box inference is nonfinite; revise the lag or residual sample")
    return {
        "method": "weak_efficiency", "null_hypothesis": "No diagnostic residual autocorrelation through lags",
        "h": int(h), "ma_order": q, "lags": int(lags), "statistic": statistic,
        "distribution": "chi2", "pvalue": pvalue, "df": int(lags) - q, "model_df": q,
        "alpha": float(alpha), "reject_null": bool(pvalue < alpha), "nobs": n,
        "diagnostic_nobs": len(diagnostic), "ma_coefficients": coefficients,
        "ma_standard_errors": standard_errors, "estimation": estimation,
    }


def orthogonality_test(actual, forecast, information, h=1, cov_type="HAC",
                       maxlags=None, alpha=0.05):
    r"""Jointly test whether origin-available information predicts errors.

    Parameters
    ----------
    actual, forecast : numpy.ndarray, sequence, or pandas.Series
        Nonempty one-dimensional finite real observations and forecasts, paired
        in order with errors e_(t+h|t) = actual - forecast. Lengths must match;
        two Series must have identical indexes. No observations are dropped.
    information : numpy.ndarray, sequence, pandas.Series, or pandas.DataFrame
        One or several finite real information variables. A one-dimensional
        input is one variable; a matrix has observations in rows and variables
        in columns. Rows must already correspond to the same forecast-error
        pairs. If information and either actual/forecast have pandas indexes,
        these must match in the same order. No automatic alignment or shift is
        performed. Caller must verify every value was available at the forecast
        origin, not merely at the target date: look-ahead bias invalidates the
        interpretation. Names are preserved for named Series/DataFrames;
        unnamed variables use information_1, information_2, etc. Names must be
        unique and must not equal the reserved parameter name "intercept".
        Do not include an intercept column; the function supplies it.
    h : int, default 1
        Positive forecast horizon, recorded as metadata. h does not shift rows
        or change regressors. Overlapping forecasts can produce serially
        dependent regression disturbances.
    cov_type : {"HAC", "nonrobust"}, default "HAC"
        HAC uses Bartlett/Newey-West covariance without a small-sample
        multiplier, an asymptotic chi-square joint Wald test with k restrictions,
        and normal marginal confidence intervals. nonrobust uses OLS covariance,
        a classical F(k,n-k) joint test, and Student t marginal intervals.
        Here k counts the intercept plus all information coefficients.
    maxlags : int or None, default None
        HAC bandwidth, using the Stage 3 convention:
        min(n-1, floor(4*(n/100)**(2/9))) when None. Explicit bandwidth must be
        a nonnegative integer below n. nonrobust requires None. The automatic
        rule is not increased based on h; for overlap consider at least h-1
        lags and potentially more for additional serial dependence. Smaller
        bandwidths remain allowed and are flagged in the return metadata.
    alpha : float, default 0.05
        Significance level strictly between zero and one. Confidence coverage
        is 1-alpha and rejection occurs when pvalue < alpha.

    Returns
    -------
    dict
        ``method`` ("orthogonality"), ``null_hypothesis``, ``h``, ``nobs``,
        ``statistic``, ``distribution`` ("chi2" or "F"), ``pvalue``, ``alpha``,
        ``reject_null``, ``cov_type``, ``maxlags``, ``n_restrictions`` (k),
        ``df_resid`` (n-k), ``df`` (k for chi-square, otherwise None),
        ``df_num``/``df_denom`` (k/n-k for F, otherwise None).
        ``parameters``, ``standard_errors``, ``confidence_intervals`` are
        dictionaries keyed by "intercept" and information variable names;
        intervals are (lower, upper) tuples. ``information_names`` lists the
        input variable names in column order. ``bandwidth_covers_overlap`` is
        maxlags >= h-1 for HAC, or None for nonrobust.

    Raises
    ------
    TypeError
        For nonnumeric observations/information.
    ValueError
        For invalid options, missing/nonfinite data, index/length mismatches,
        absent information columns, n <= k, duplicate/reserved names,
        rank-deficient designs, or degenerate residual variance/covariance.

    Notes
    -----
    Estimate e_(t+h|t) = a + gamma' z_t + u_(t+h) and jointly test
    H0: a=0 and gamma=0 against at least one nonzero coefficient. This includes
    a zero-bias restriction; separate coefficient tests are not substitutes.
    Failure to reject only concerns the supplied information and functional
    form; it is not proof of forecast optimality. Predictability from omitted
    information or nonlinear relations may remain. Weak efficiency examines
    serial restrictions instead of these explicit information restrictions.

    Exact conventional F inference requires independent homoskedastic Gaussian
    disturbances and a correctly specified regression; overlapping multistep
    errors generally call for HAC. HAC inference is asymptotic, can be weak in
    small samples, and does not fix look-ahead bias, endogenous regressors,
    first-stage estimation effects, or an inadequate bandwidth. Horizon and
    availability timing must be established by the caller.

    Examples
    --------
    >>> actual = [11, 19, 32, 38, 53, 59]
    >>> forecast = [10, 20, 30, 40, 50, 60]
    >>> origin_signal = pd.Series([0, 1, 0, 1, 0, 1], name="origin_signal")
    >>> result = orthogonality_test(actual, forecast, origin_signal, maxlags=1)
    >>> result['information_names'], result['n_restrictions']
    (['origin_signal'], 2)
    >>> result = orthogonality_test(actual, forecast, origin_signal,
    ...                             cov_type="nonrobust")
    >>> result['distribution'], result['df_denom']
    ('F', 4)
    """
    from statsmodels.regression.linear_model import OLS

    _efficiency_options(h, alpha)
    if not isinstance(cov_type, str) or cov_type not in ("HAC", "nonrobust"):
        raise ValueError("cov_type must be 'HAC' or 'nonrobust'")
    if maxlags is not None:
        if (isinstance(maxlags, (bool, np.bool_)) or not isinstance(maxlags, numbers.Integral)
                or maxlags < 0):
            raise ValueError("maxlags must be None or a nonnegative integer")
        if cov_type == "nonrobust":
            raise ValueError("maxlags is only applicable to HAC; use None with nonrobust")
    errors = _efficiency_errors(actual, forecast)
    n = len(errors)
    if maxlags is not None and maxlags >= n:
        raise ValueError("HAC maxlags must be smaller than the number of observations")
    if isinstance(information, (pd.Series, pd.DataFrame)):
        for label, sample in (("actual", actual), ("forecast", forecast)):
            if isinstance(sample, pd.Series) and not sample.index.equals(information.index):
                raise ValueError(f"information and {label} indexes must match in the same order")
    if np.ma.isMaskedArray(information) and np.ma.getmaskarray(information).any():
        raise ValueError("information contains masked observations; missing values are not allowed")
    try:
        raw = np.asarray(information)
    except (TypeError, ValueError) as exc:
        raise ValueError("information must be a one- or two-dimensional numeric sample") from exc
    if raw.ndim not in (1, 2):
        raise ValueError("information must be one- or two-dimensional")
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.shape[0] != n:
        raise ValueError("information must have the same number of observations as actual and forecast")
    if raw.shape[1] == 0:
        raise ValueError("information must contain at least one variable")
    if isinstance(information, pd.DataFrame):
        names = list(information.columns)
    elif isinstance(information, pd.Series) and information.name is not None:
        names = [information.name]
    else:
        names = [f"information_{j+1}" for j in range(raw.shape[1])]
    if len(set(names)) != len(names) or "intercept" in names:
        raise ValueError("information variable names must be unique and not equal 'intercept'")
    # Reuse the existing numeric validation column by column without alignment.
    columns = [_accuracy_inputs(raw[:, j], errors)[0] for j in range(raw.shape[1])]
    x = np.column_stack([np.ones(n), *columns])
    k = x.shape[1]
    if n <= k:
        raise ValueError("Orthogonality inference requires more observations than regression parameters")
    if np.linalg.matrix_rank(x) != k:
        raise ValueError("Orthogonality design is rank-deficient; exclude constant or redundant information variables")
    fit = OLS(errors, x, missing="raise").fit(use_t=True)
    if np.linalg.norm(fit.resid) <= n * np.finfo(float).eps * max(1.0, np.linalg.norm(errors)):
        raise ValueError("Residual variance is zero or numerically degenerate; orthogonality inference is undefined")
    bandwidth = None
    if cov_type == "HAC":
        bandwidth = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9)))) if maxlags is None else int(maxlags)
        fit = fit.get_robustcov_results(cov_type="HAC", use_t=False, maxlags=bandwidth,
                                        kernel="bartlett", use_correction=False)
    covariance = np.asarray(fit.cov_params())
    if not np.isfinite(covariance).all() or np.linalg.matrix_rank(covariance) != k:
        raise ValueError("Parameter covariance is nonfinite or singular; orthogonality inference is undefined")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Parameter covariance is not positive definite; orthogonality inference is undefined") from exc
    test = fit.wald_test(np.eye(k), use_f=cov_type == "nonrobust", scalar=True)
    statistic, pvalue = float(test.statistic), float(test.pvalue)
    intervals = np.asarray(fit.conf_int(alpha=float(alpha)))
    if not np.isfinite([statistic, pvalue]).all() or not np.isfinite(intervals).all():
        raise ValueError("Nonfinite inference results; rescale observations or revise the design")
    parameters = ["intercept", *names]
    classical = cov_type == "nonrobust"
    return {
        "method": "orthogonality", "null_hypothesis": "intercept = 0 and all information coefficients = 0",
        "h": int(h), "nobs": n, "statistic": statistic, "distribution": "F" if classical else "chi2",
        "pvalue": pvalue, "alpha": float(alpha), "reject_null": bool(pvalue < alpha),
        "cov_type": cov_type, "maxlags": bandwidth, "n_restrictions": k,
        "df_resid": n - k, "df": None if classical else k,
        "df_num": k if classical else None, "df_denom": n - k if classical else None,
        "parameters": dict(zip(parameters, map(float, fit.params))),
        "standard_errors": dict(zip(parameters, map(float, fit.bse))),
        "confidence_intervals": dict(zip(parameters, [tuple(map(float, row)) for row in intervals])),
        "information_names": names,
        "bandwidth_covers_overlap": None if classical else bandwidth >= int(h) - 1,
    }


def mgn_test(actual, forecast1, forecast2, cov_type="HAC", maxlags=None,
             alpha=0.05):
    r"""Compare two forecast-error variances through the MGN regression.

    Parameters
    ----------
    actual : numpy.ndarray, sequence, or pandas.Series
        Nonempty, one-dimensional finite real evaluation observations.
    forecast1, forecast2 : numpy.ndarray, sequence, or pandas.Series
        Forecasts for the same observations, with errors e1 = actual-forecast1
        and e2 = actual-forecast2. All lengths must match. Any two pandas Series
        among the three inputs must have identical indexes in the same order.
        Mixed labeled/unlabeled inputs pair positionally. No observations are
        truncated, reordered, aligned, or discarded. At least three pairs are
        required, with nondegenerate sum/difference and regression residuals.
    cov_type : {"HAC", "nonrobust"}, default "HAC"
        HAC uses Newey-West/Bartlett covariance without a small-sample
        multiplier, asymptotic normal slope inference, and normal marginal
        coefficient intervals. nonrobust uses ordinary OLS covariance,
        Student t slope inference, and t intervals with N-2 degrees of freedom.
    maxlags : int or None, default None
        HAC bandwidth q. None selects min(N-1, floor(4*(N/100)**(2/9))),
        as in Stages 3 and 4. Explicit bandwidth must be a nonnegative integer
        below N; booleans are rejected. nonrobust requires None. Observation
        order must represent consecutive forecast evaluation periods. The
        caller must choose a suitable bandwidth for overlap/serial dependence.
    alpha : float, default 0.05
        Significance level strictly between zero and one. Coefficient interval
        coverage is 1-alpha; rejection occurs when pvalue < alpha.
        This argument is distinct from the regression intercept.

    Returns
    -------
    dict
        ``method`` ("mgn"), ``null_hypothesis`` ("slope = 0"), ``nobs``,
        ``statistic`` (signed slope divided by its standard error),
        ``distribution`` ("normal" or "t"), ``df`` (None or N-2),
        ``df_resid`` (N-2), ``n_restrictions`` (1), ``pvalue`` (two-sided),
        ``alpha``, ``reject_null``, ``cov_type``, ``maxlags`` (selected bandwidth
        or None), and ``correlation`` (sample correlation of u and v).
        ``parameters`` and ``standard_errors`` are float dictionaries keyed
        by "intercept" and "slope". ``confidence_intervals`` maps those names
        to marginal (lower, upper) tuples. ``mean_errors`` and ``mse`` are
        descriptive float dictionaries keyed by "model1" and "model2".
        ``mse_difference`` is sample MSE(model1)-MSE(model2); a positive value
        means model2 has smaller sample MSE. ``msfe_interpretation_note`` states
        the required zero-population-mean assumption. No additional hypothesis
        test is constructed from these descriptive MSE summaries.

    Raises
    ------
    TypeError
        For nonnumeric observations.
    ValueError
        For invalid options/samples, inconsistent indexes/lengths, insufficient
        observations, constant or numerically degenerate u/v, perfect or
        near-perfect correlation, rank deficiency, zero residual variance,
        singular/invalid covariance, or overflow/nonfinite calculations.

    Notes
    -----
    Define u_t=e1_t+e2_t and v_t=e1_t-e2_t. Estimate
    u_t = a + b*v_t + disturbance_t and test H0: b=0 against H1: b!=0.
    The intercept is unrestricted. With positive Var(v), this tests
    Cov(u,v)=Var(e1)-Var(e2)=0. Only if BOTH errors have zero population means
    does this equal the null of equal population MSFE, E(e1**2)=E(e2**2).
    Otherwise the intercept regression tests equality of centered error
    variances, not equality of expected squared losses. Small sample mean
    errors or failed unbiasedness rejections do not prove the population
    assumption. Direct loss-differential inference belongs to Diebold-Mariano.

    Conventional inference reproduces the classical correlation statistic
    t = r*sqrt((N-2)/(1-r**2)), r=Corr(u,v), using Student t(N-2).
    Exact conventional reference inference requires independent observations
    and suitable normal/homoskedastic regression assumptions. HAC allows
    heteroskedasticity and weak serial dependence under regularity conditions
    but is asymptotic and may be unreliable in small samples. Neither choice
    automatically resolves first-stage estimation effects or bandwidth choice.

    For numerical safeguards, reject 1-r**2 <= 1e-12 (perfect/near-perfect
    collinearity); correlation is computed from scaled centered vectors to
    avoid unnecessary overflow. Numerically rank-deficient designs or
    degenerate covariance are also rejected rather than returning unstable
    p-values. Rejection concerns the stated covariance restriction. Failure
    to reject does not establish equal forecasting performance or optimality.
    Swapping models flips the statistic's sign but preserves its two-sided
    p-value. A statistic's performance interpretation requires the assumptions
    above, not merely the ordering of descriptive sample MSEs.

    Examples
    --------
    >>> actual = [10, 20, 30, 40, 50]
    >>> forecast1 = [9, 22, 27, 39, 48]
    >>> forecast2 = [12, 19, 31, 37, 51]
    >>> result = mgn_test(actual, forecast1, forecast2, cov_type="nonrobust")
    >>> result['distribution'], result['df']
    ('t', 3)
    >>> round(result['mse_difference'], 6)
    0.6
    >>> result = mgn_test(actual, forecast1, forecast2, maxlags=1)
    >>> result['distribution'], result['maxlags']
    ('normal', 1)
    """
    from statsmodels.regression.linear_model import OLS

    if not isinstance(cov_type, str) or cov_type not in ("HAC", "nonrobust"):
        raise ValueError("cov_type must be 'HAC' or 'nonrobust'")
    _efficiency_options(1, alpha)
    if maxlags is not None:
        if (isinstance(maxlags, (bool, np.bool_)) or not isinstance(maxlags, numbers.Integral)
                or maxlags < 0):
            raise ValueError("maxlags must be None or a nonnegative integer")
        if cov_type == "nonrobust":
            raise ValueError("maxlags is only applicable to HAC; use None with nonrobust")
    actual_values, first = _accuracy_inputs(actual, forecast1)
    _, second = _accuracy_inputs(actual, forecast2)
    # Check forecast indexes even when actual is an unlabeled array.
    _accuracy_inputs(forecast1, forecast2)
    n = len(actual_values)
    if n < 3:
        raise ValueError("MGN regression inference requires at least three observations")
    if maxlags is not None and maxlags >= n:
        raise ValueError("HAC maxlags must be smaller than the number of observations")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            e1, e2 = actual_values - first, actual_values - second
            u, v = e1 + e2, e1 - e2
            mean_errors = {"model1": float(e1.mean()), "model2": float(e2.mean())}
            squared_errors = {"model1": float(np.mean(np.square(e1))),
                              "model2": float(np.mean(np.square(e2)))}
            mse_difference = squared_errors["model1"] - squared_errors["model2"]
            _efficiency_variation(u, "MGN error sum u")
            _efficiency_variation(v, "MGN error difference v")
            centered_u, centered_v = u - u.mean(), v - v.mean()
            scaled_u = centered_u / np.max(np.abs(centered_u))
            scaled_v = centered_v / np.max(np.abs(centered_v))
            correlation = float((scaled_u @ scaled_v) /
                                (np.linalg.norm(scaled_u) * np.linalg.norm(scaled_v)))
        except FloatingPointError as exc:
            raise ValueError("MGN calculations overflow or are numerically invalid; rescale observations") from exc
    if not np.isfinite(mse_difference):
        raise ValueError("MSE difference is nonfinite; rescale observations")
    correlation = float(np.clip(correlation, -1.0, 1.0))
    if 1 - correlation ** 2 <= 1e-12:
        raise ValueError("MGN sum and difference are perfectly or near-perfectly collinear; inference is undefined")
    x = np.column_stack([np.ones(n), v])
    if np.linalg.matrix_rank(x) != 2:
        raise ValueError("MGN regression design is numerically rank-deficient; rescale or revise the errors")
    fit = OLS(u, x, missing="raise").fit(use_t=True)
    if np.linalg.norm(fit.resid) <= n * np.finfo(float).eps * max(1.0, np.linalg.norm(u)):
        raise ValueError("MGN residual variance is zero or numerically degenerate")
    bandwidth = None
    classical = cov_type == "nonrobust"
    if not classical:
        bandwidth = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9)))) if maxlags is None else int(maxlags)
        fit = fit.get_robustcov_results(cov_type="HAC", use_t=False, maxlags=bandwidth,
                                        kernel="bartlett", use_correction=False)
    covariance = np.asarray(fit.cov_params())
    if not np.isfinite(covariance).all() or np.linalg.matrix_rank(covariance) != 2:
        raise ValueError("MGN parameter covariance is nonfinite or singular; inference is undefined")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("MGN parameter covariance is not positive definite; inference is undefined") from exc
    test = fit.t_test(np.array([[0.0, 1.0]]), use_t=classical)
    statistic, pvalue = float(np.asarray(test.statistic).item()), float(np.asarray(test.pvalue).item())
    intervals = np.asarray(fit.conf_int(alpha=float(alpha)))
    if not np.isfinite([statistic, pvalue, *fit.params, *fit.bse]).all() or not np.isfinite(intervals).all():
        raise ValueError("MGN inference results are nonfinite; rescale observations or revise the design")
    names = ["intercept", "slope"]
    return {
        "method": "mgn", "null_hypothesis": "slope = 0", "nobs": n,
        "statistic": statistic, "distribution": "t" if classical else "normal",
        "df": n - 2 if classical else None, "df_resid": n - 2, "n_restrictions": 1,
        "pvalue": pvalue, "alpha": float(alpha), "reject_null": bool(pvalue < alpha),
        "cov_type": cov_type, "maxlags": bandwidth, "correlation": correlation,
        "parameters": dict(zip(names, map(float, fit.params))),
        "standard_errors": dict(zip(names, map(float, fit.bse))),
        "confidence_intervals": dict(zip(names, [tuple(map(float, row)) for row in intervals])),
        "mean_errors": mean_errors, "mse": squared_errors, "mse_difference": mse_difference,
        "msfe_interpretation_note": "Equal-MSFE interpretation requires both errors to have zero population means; otherwise this tests a covariance (centered-variance) restriction. Sample MSE summaries are descriptive only.",
    }


def dm_test(actual, forecast1, forecast2, loss="squared", h=1,
            alternative="two-sided", maxlags=None, hln=False, alpha=0.05):
    r"""Test equality of expected forecast losses with direct Bartlett HAC DM.

    Parameters
    ----------
    actual, forecast1, forecast2 : array-like or pandas.Series
        Finite real, one-dimensional observations for identical evaluation
        periods, with at least two observations. Any pair of Series must have
        matching indexes in the same order. Other inputs pair positionally;
        missing observations are never removed or silently aligned.
    loss : {"squared", "absolute"}, default "squared"
        Loss L(e) = e**2 or abs(e), with e = actual - forecast.
    h : int, default 1
        Positive forecast horizon. Metadata only: does not shift observations
        or choose the bandwidth. HLN requires h < sample size.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative E(d) != 0, E(d) < 0, or E(d) > 0, respectively, for
        d = L(e1) - L(e2). Negative values favor model 1; positive favor model 2.
    maxlags : int or None, default None
        Bartlett bandwidth q, from zero through N-1. None selects
        min(N-1, floor(4*(N/100)**(2/9))). This practical default is not
        universally optimal. For overlapping h-step forecasts, h-1 is a
        conventional choice under suitable assumptions; additional dependence
        may require more lags. Explicit bandwidths are never overridden.
    hln : bool, default False
        Apply Harvey–Leybourne–Newbold correction and Student t(N-1)
        inference when True; otherwise use conventional asymptotic normal DM.
    alpha : float, default 0.05
        Significance level strictly between zero and one.

    Returns
    -------
    dict
        ``n``, ``nobs``, ``h``, ``loss``, ``alternative``, ``mean_loss1``,
        ``mean_loss2``, ``mean_loss_diff``, ``long_run_variance``, ``se``,
        ``dm_statistic`` (uncorrected), ``statistic`` (used for inference),
        ``pvalue``, ``maxlags``, ``hln``, ``hln_factor`` (None if disabled),
        ``df`` (None or N-1), ``distribution`` ("normal" or "t"), ``alpha``,
        ``reject_null``, ``method`` and ``null_hypothesis``.

    Raises
    ------
    TypeError
        For nonnumeric observations.
    ValueError
        For invalid options, mismatched lengths/indexes, missing/nonfinite
        observations, fewer than two observations, invalid HLN horizons,
        overflow, or zero/nonpositive/numerically degenerate long-run variance.

    Notes
    -----
    Compute centered autocovariances gamma_k = sum((d_t-dbar)*
    (d_{t-k}-dbar))/N and Omega = gamma_0 + 2*sum((1-k/(q+1))*gamma_k).
    The reported conventional DM is computed directly as dbar/sqrt(Omega/N),
    with no small-sample covariance multiplier. It equals the intercept-only
    OLS HAC statistic with identical normalization; regression is not used here.

    HLN multiplies DM by sqrt((N+1-2*h+h*(h-1)/N)/N), without changing losses,
    Omega or SE. It is a finite-sample adjustment developed under particular
    forecasting assumptions, not exact inference for arbitrary dependence or
    heteroskedasticity. HAC inference requires suitable stationarity, moments
    and weak dependence; neither option resolves first-stage estimation effects.
    Short samples can give unreliable inference. Unlike MGN's covariance
    restriction, DM directly tests expected loss equality without requiring
    zero-mean forecast errors. Nonrejection does not prove equal accuracy or
    optimality; observed loss signs alone do not establish significance.

    Examples
    --------
    >>> result = dm_test([10, 20, 30, 40, 50], [9, 22, 27, 39, 48],
    ...                  [12, 19, 31, 37, 51], maxlags=1)
    >>> round(result['mean_loss_diff'], 6)
    0.6
    >>> result['distribution'], result['hln_factor']
    ('normal', None)
    """
    from scipy.stats import t

    _efficiency_options(h, alpha)
    if not isinstance(loss, str) or loss not in ("squared", "absolute"):
        raise ValueError("loss must be 'squared' or 'absolute'")
    if not isinstance(alternative, str) or alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    if not isinstance(hln, (bool, np.bool_)):
        raise ValueError("hln must be a boolean")
    if maxlags is not None and (isinstance(maxlags, (bool, np.bool_))
            or not isinstance(maxlags, numbers.Integral) or maxlags < 0):
        raise ValueError("maxlags must be None or a nonnegative integer")
    # Preserve type information before NumPy coerces mixed bool/float sequences.
    for name, values in (("actual", actual), ("forecast1", forecast1), ("forecast2", forecast2)):
        raw_objects = np.asarray(values, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in raw_objects.flat):
            raise TypeError(f"{name} must contain real numeric observations, not booleans")
    y, f1 = _accuracy_inputs(actual, forecast1)
    _, f2 = _accuracy_inputs(actual, forecast2)
    _accuracy_inputs(forecast1, forecast2)
    n = len(y)
    if n < 2:
        raise ValueError("DM inference requires at least two observations")
    if maxlags is not None and maxlags >= n:
        raise ValueError("HAC maxlags must be smaller than the number of observations")
    if hln and h >= n:
        raise ValueError("HLN requires forecast horizon h smaller than the sample size")
    q = min(n-1, int(np.floor(4*(n/100)**(2/9)))) if maxlags is None else int(maxlags)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            e1, e2 = y-f1, y-f2
            l1 = np.square(e1) if loss == "squared" else np.abs(e1)
            l2 = np.square(e2) if loss == "squared" else np.abs(e2)
            d = l1-l2
            mean = float(d.mean())
            centered = d-mean
            scale = float(np.max(np.abs(d)))
            if scale == 0 or np.max(np.abs(centered))/scale <= n*np.finfo(float).eps:
                raise ValueError("Loss differential variance is zero or numerically degenerate")
            gamma0 = float(centered @ centered / n)
            omega = gamma0
            for k in range(1, q+1):
                omega += 2*(1-k/(q+1))*float(centered[k:] @ centered[:-k]/n)
            if not np.isfinite(omega) or omega <= np.finfo(float).eps*gamma0 or gamma0 <= 0:
                raise ValueError("HAC long-run variance is nonpositive, nonfinite or numerically degenerate")
            se = float(np.sqrt(omega/n))
            dm = mean/se
            means = float(l1.mean()), float(l2.mean())
        except FloatingPointError as exc:
            raise ValueError("DM calculations overflow or are numerically invalid; rescale observations") from exc
    factor = float(np.sqrt((n+1-2*h+h*(h-1)/n)/n)) if hln else None
    if hln and (not np.isfinite(factor) or factor <= 0):
        raise ValueError("HLN correction factor must be finite and strictly positive")
    statistic = dm*factor if hln else dm
    reference = t(n-1) if hln else norm
    pvalue = float(2*reference.sf(abs(statistic)) if alternative == "two-sided"
                   else reference.cdf(statistic) if alternative == "less"
                   else reference.sf(statistic))
    if not np.isfinite([mean, omega, se, dm, statistic, pvalue, *means]).all() or se <= 0:
        raise ValueError("DM inference is nonfinite or numerically degenerate; rescale observations")
    return {"method": "dm", "null_hypothesis": "Expected loss differential = 0",
            "n": n, "nobs": n, "h": int(h), "loss": loss, "alternative": alternative,
            "mean_loss1": means[0], "mean_loss2": means[1], "mean_loss_diff": mean,
            "long_run_variance": omega, "se": se, "dm_statistic": dm,
            "statistic": statistic, "pvalue": pvalue, "maxlags": q,
            "hln": bool(hln), "hln_factor": factor, "df": n-1 if hln else None,
            "distribution": "t" if hln else "normal", "alpha": float(alpha),
            "reject_null": bool(pvalue < alpha)}
