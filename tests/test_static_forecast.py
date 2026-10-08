"""Run with: python -m unittest discover -s tests -v."""
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

from forecasting_tools import static_forecast


class StaticForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(738)
        innovations = rng.normal(size=100)
        cls.stationary = np.zeros(100)
        for t in range(1, 100):
            cls.stationary[t] = .55 * cls.stationary[t - 1] + innovations[t] + .3 * innovations[t - 1]
        cls.dates = pd.date_range('2020-01-01', periods=100, freq='D')

    def check_filter(self, order, index, dataframe=False, array=False, trend=None):
        data = self.stationary if order[1] == 0 else self.stationary.cumsum()
        series = pd.Series(data, index=index, name='training')
        training = series.iloc[:80]
        if dataframe:
            training = training.to_frame()
        if array:
            training = training.to_numpy()
        kwargs = {} if trend is None else {'trend': trend}
        fitted = ARIMA(training, order=order, **kwargs).fit()
        params = fitted.params.copy()
        actual = series.iloc[80:].rename('different evaluation name')
        output = static_forecast(fitted, actual, level=.9)
        # Independent oracle: filter the entire history once with training params.
        full = fitted.model.clone(series.to_numpy() if array else series)
        full.ssm.initialization = fitted.model.ssm.initialization
        filtered = full.filter(params)
        oracle = filtered.get_prediction(start=80, end=99, dynamic=False)
        np.testing.assert_allclose(output.Forecast, np.asarray(oracle.predicted_mean), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(output.SE, np.asarray(oracle.se_mean), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(output.Error, actual.to_numpy() - output.Forecast)
        np.testing.assert_allclose(output.Lower, output.Forecast - norm.ppf(.95) * output.SE)
        np.testing.assert_allclose(output.Upper, output.Forecast + norm.ppf(.95) * output.SE)
        pd.testing.assert_index_equal(output.index, actual.index)
        np.testing.assert_array_equal(fitted.params, params)
        self.assertEqual(fitted.nobs, 80)
        # Later observations must not affect earlier forecasts.
        altered = actual.copy()
        altered.iloc[5:] += 50
        np.testing.assert_allclose(static_forecast(fitted, altered).Forecast.iloc[:6], output.Forecast.iloc[:6])
        return fitted, actual

    def test_arima_filter_equivalence(self):
        for order in [(1, 0, 1), (1, 1, 1)]:
            for kind in ['datetime', 'period', 'dataframe', 'array', 'range']:
                with self.subTest(order=order, kind=kind):
                    index = self.dates.to_period('D') if kind == 'period' else pd.RangeIndex(100) if kind == 'range' else self.dates
                    self.check_filter(order, index, dataframe=kind == 'dataframe', array=kind == 'array')

    def test_integrated_drift(self):
        self.check_filter((1, 1, 1), self.dates, trend='t')

    def test_validation_and_empty(self):
        fit = ARIMA(pd.Series(self.stationary[:80], index=self.dates[:80]), order=(1, 0, 1)).fit()
        actual = pd.Series(self.stationary[80:], index=self.dates[80:])
        for level in [0, 1, -1, np.nan, np.inf, True, '95%']:
            with self.subTest(level=level), self.assertRaises(ValueError):
                static_forecast(fit, actual, level)
        for bad in [[np.nan], [np.inf], [[1, 2]], [1+2j], [True], ['invalid']]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                static_forecast(fit, bad)
        for bad in [actual.iloc[::-1], actual.iloc[1:], pd.Series([1, 2], index=[self.dates[80]] * 2)]:
            with self.assertRaises(ValueError):
                static_forecast(fit, bad)
        with self.assertRaises(TypeError):
            static_forecast(fit, actual.to_frame())
        with self.assertRaises(TypeError):
            static_forecast(object(), actual)
        empty = static_forecast(fit, actual.iloc[:0])
        self.assertEqual(empty.shape, (0, 6))
        pd.testing.assert_index_equal(empty.index, actual.iloc[:0].index)
        inferred = static_forecast(fit, actual.to_numpy())
        pd.testing.assert_index_equal(inferred.index, actual.index)

    def test_reject_exog_seasonal_and_irregular_dates(self):
        exog = np.arange(80.)[:, None]
        fit = ARIMA(self.stationary[:80], exog=exog, order=(1, 0, 1)).fit()
        with self.assertRaises(ValueError):
            static_forecast(fit, [1])
        fit = ARIMA(self.stationary[:80], order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)).fit()
        with self.assertRaises(ValueError):
            static_forecast(fit, [1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = ARIMA(pd.Series(self.stationary[:3], index=self.dates[[0, 2, 5]]), order=(0, 0, 0)).fit()
        with self.assertRaises(ValueError):
            static_forecast(fit, [1])


if __name__ == '__main__':
    unittest.main()
