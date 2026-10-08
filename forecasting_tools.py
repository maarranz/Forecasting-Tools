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
