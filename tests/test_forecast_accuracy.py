"""Independent numerical and validation checks for accuracy measures."""
import math
import unittest

import numpy as np
import pandas as pd

from forecasting_tools import me, mae, mse, rmse, mape, forecast_accuracy


STATISTICS = (me, mae, mse, rmse, mape)
ALL_FUNCTIONS = STATISTICS + (forecast_accuracy,)
LABELS = ['ME', 'MAE', 'MSE', 'RMSE', 'MAPE']


class ForecastAccuracyTests(unittest.TestCase):
    def assert_statistics(self, actual, forecast, expected):
        for function, value in zip(STATISTICS, expected):
            with self.subTest(function=function.__name__):
                result = function(actual, forecast)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, value)
        combined = forecast_accuracy(actual, forecast)
        self.assertIsInstance(combined, pd.Series)
        self.assertEqual(combined.index.tolist(), LABELS)
        np.testing.assert_allclose(combined.to_numpy(), expected)
        np.testing.assert_allclose(
            combined.to_numpy(), [function(actual, forecast) for function in STATISTICS]
        )

    def test_independent_numerical_example_and_input_types(self):
        # Errors: 2, -3, 4. Sum=3; absolute sum=9; squared sum=29.
        # Percentage terms: 20, 15, 40/3; their mean is 145/9.
        expected = [1, 3, 29/3, math.sqrt(29/3), 145/9]
        for constructor in (list, tuple, np.array, pd.Series):
            with self.subTest(input_type=constructor.__name__):
                self.assert_statistics(constructor([10, 20, 30]), constructor([8, 23, 26]), expected)
        dates = pd.date_range('2026-01-01', periods=3, freq='D')
        actual = pd.Series([10, 20, 30], index=dates, name='actual')
        forecast = pd.Series([8, 23, 26], index=dates, name='forecast')
        self.assert_statistics(actual, forecast, expected)
        # Mixed labeled/unlabeled inputs pair positionally, without alignment.
        self.assert_statistics(actual, [8, 23, 26], expected)
        self.assert_statistics(np.array([10, 20, 30]), forecast, expected)
        nullable = pd.Series([10, 20, 30], dtype='Int64')
        self.assert_statistics(nullable, pd.Series([8, 23, 26], dtype='Float64'), expected)

    def test_perfect_forecasts(self):
        self.assert_statistics([1, -2, 3], [1, -2, 3], [0] * 5)

    def test_signed_bias_and_error_magnitude(self):
        self.assert_statistics([10, 20], [8, 16], [3, 3, 10, math.sqrt(10), 20])
        self.assert_statistics([10, 20], [12, 24], [-3, 3, 10, math.sqrt(10), 20])
        self.assert_statistics([10, 20], [8, 22], [0, 2, 4, 2, 15])
        self.assert_statistics([-10, -20], [-8, -24], [1, 3, 10, math.sqrt(10), 20])

    def test_single_observation(self):
        self.assert_statistics([4], [3], [1, 1, 1, 1, 25])

    def test_lengths_and_empty_samples(self):
        for function in ALL_FUNCTIONS:
            for actual, forecast in (([1, 2], [1]), ([], []), ([], [1]), ([1], [])):
                with self.subTest(function=function.__name__, actual=actual, forecast=forecast):
                    with self.assertRaisesRegex(ValueError, 'length|nonempty'):
                        function(actual, forecast)

    def test_indexes_must_match_in_order(self):
        actual = pd.Series([10, 20], index=['a', 'b'])
        for forecast in (pd.Series([8, 16], index=['b', 'a']),
                         pd.Series([8, 16], index=['a', 'c'])):
            for function in ALL_FUNCTIONS:
                with self.subTest(function=function.__name__, index=forecast.index):
                    with self.assertRaisesRegex(ValueError, 'indexes.*same order'):
                        function(actual, forecast)
        # Matching descending indexes are valid: preserve supplied pair order.
        self.assert_statistics(pd.Series([10, 20], index=['b', 'a']),
                               pd.Series([8, 16], index=['b', 'a']),
                               [3, 3, 10, math.sqrt(10), 20])

    def test_missing_and_infinite_values_in_either_input(self):
        bad_samples = ([1, np.nan], [1, np.inf], [1, -np.inf], [1, None],
                       pd.Series([1, pd.NA], dtype='Float64'),
                       np.ma.array([1, 2], mask=[False, True]))
        for function in ALL_FUNCTIONS:
            for bad in bad_samples:
                for actual, forecast in ((bad, [1, 2]), ([1, 2], bad)):
                    with self.subTest(function=function.__name__, actual=actual, forecast=forecast):
                        with self.assertRaisesRegex(ValueError, 'finite|missing|masked'):
                            function(actual, forecast)

    def test_nonnumeric_and_nonvector_inputs(self):
        for function in ALL_FUNCTIONS:
            for bad in (['1', '2'], ['one', 'two'], [True, False], [1+2j, 2],
                        np.array([object(), 2], dtype=object),
                        pd.date_range('2026-01-01', periods=2)):
                for actual, forecast in ((bad, [1, 2]), ([1, 2], bad)):
                    with self.subTest(function=function.__name__, bad=bad):
                        with self.assertRaisesRegex(TypeError, 'numeric'):
                            function(actual, forecast)
            for bad in (1, [[1, 2]], [[1], [2]], pd.DataFrame({'a': [1, 2]})):
                for actual, forecast in ((bad, [1, 2]), ([1, 2], bad)):
                    with self.subTest(function=function.__name__, bad=bad):
                        with self.assertRaises((TypeError, ValueError)):
                            function(actual, forecast)

    def test_mape_zero_actuals(self):
        for function in (mape, forecast_accuracy):
            for actual in ([0, 2], [-0.0, 2], [0]):
                with self.subTest(function=function.__name__, actual=actual):
                    with self.assertRaisesRegex(ValueError, 'MAPE.*undefined.*zero'):
                        function(actual, [1] * len(actual))
        # Only percentage measures reject zero actuals; zero forecasts are valid.
        for function in STATISTICS[:4]:
            self.assertEqual(function([0, 2], [0, 2]), 0)
        self.assertEqual(mape([1, 2], [0, 0]), 100)
        self.assertGreater(mape([0.001], [1]), 100)

    def test_inputs_are_unchanged(self):
        actual = pd.Series([10., 20., 30.], index=['c', 'a', 'b'])
        forecast = pd.Series([8., 23., 26.], index=actual.index)
        before_actual, before_forecast = actual.copy(), forecast.copy()
        for function in ALL_FUNCTIONS:
            function(actual, forecast)
        pd.testing.assert_series_equal(actual, before_actual)
        pd.testing.assert_series_equal(forecast, before_forecast)


if __name__ == '__main__':
    unittest.main()
