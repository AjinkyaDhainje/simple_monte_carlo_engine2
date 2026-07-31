"""Tests for pricing, batching, component combinations, and UI data limits."""

import math
import unittest

import matplotlib.pyplot as plt
import numpy as np

from monte_carlo import (
    DISCRETIZATIONS,
    MODELS,
    MonteCarloEngine,
    PAYOFFS,
    SAMPLERS,
    SimulationInputs,
    SimulationManager,
)
from monte_carlo.charts import path_chart


class EngineTests(unittest.TestCase):
    def test_every_component_combination_runs(self):
        for model in MODELS:
            for discretization in DISCRETIZATIONS:
                for payoff in PAYOFFS:
                    for option_type in ("Call", "Put"):
                        for sampling in SAMPLERS:
                            with self.subTest(
                                model=model,
                                discretization=discretization,
                                payoff=payoff,
                                option_type=option_type,
                                sampling=sampling,
                            ):
                                inputs = SimulationInputs(
                                    model=model,
                                    discretization=discretization,
                                    payoff=payoff,
                                    option_type=option_type,
                                    sampling=sampling,
                                    num_paths=128,
                                    num_steps=8,
                                )
                                result = SimulationManager().run(inputs)
                                self.assertEqual(result.display_paths.shape, (128, 9))
                                self.assertTrue(np.isfinite(result.option_price))

    def test_asian_payoff_uses_selected_final_days(self):
        payoff = PAYOFFS["Asian"]
        inputs = SimulationInputs(
            payoff="Asian", strike=10.0, num_steps=12, averaging_days=90.0
        )
        paths = np.arange(13, dtype=float).reshape(1, 13)
        np.testing.assert_allclose(payoff.calculate(paths, inputs), [1.0])

    def test_unselected_specific_inputs_are_ignored(self):
        inputs = SimulationInputs(
            model="Geometric Brownian Motion",
            payoff="Vanilla",
            jump_intensity=-999.0,
            jump_volatility=-999.0,
            mean_reversion=-999.0,
            second_mean_reversion=-999.0,
            second_volatility=-999.0,
            factor_correlation=-999.0,
            levy_variance=-999.0,
            averaging_days=-999.0,
            num_paths=128,
            num_steps=4,
        )
        result = SimulationManager().run(inputs)
        self.assertTrue(np.isfinite(result.option_price))

    def test_engine_returns_one_path_set_without_a_generator(self):
        inputs = SimulationInputs(num_paths=8, num_steps=4)
        paths = MonteCarloEngine(inputs).generate_paths(8)
        self.assertIsInstance(paths, np.ndarray)
        self.assertEqual(paths.shape, (8, 5))

    def test_antithetic_paths_use_opposite_normal_draws(self):
        inputs = SimulationInputs(
            model="Arithmetic Brownian Motion",
            discretization="Euler",
            risk_free_rate=0.0,
            num_paths=8,
            num_steps=1,
        )
        paths = MonteCarloEngine(inputs).generate_paths(8)
        pair_sums = paths[0::2, -1] + paths[1::2, -1]
        np.testing.assert_allclose(pair_sums, 2.0 * inputs.start_price)

    def test_all_paths_price_but_only_10_000_paths_are_kept_for_ui(self):
        inputs = SimulationInputs(num_paths=10_017, num_steps=4)
        result = SimulationManager().run(inputs)

        self.assertEqual(result.display_paths.shape, (10_000, 5))
        self.assertEqual(result.terminal_prices.shape, (10_017,))
        self.assertEqual(result.discounted_payoffs.shape, (10_017,))

        figure = path_chart(result.display_paths, result.time_grid, inputs.strike)
        self.assertEqual(len(figure.axes[0].lines), 101)
        plt.close(figure)

    def test_one_million_paths_run_in_batches(self):
        inputs = SimulationInputs(num_paths=1_000_000, num_steps=1)
        result = SimulationManager().run(inputs)
        self.assertEqual(result.display_paths.shape, (10_000, 2))
        self.assertEqual(len(result.terminal_prices), 1_000_000)
        self.assertEqual(len(result.discounted_payoffs), 1_000_000)

    def test_gbm_price_is_close_to_black_scholes(self):
        inputs = SimulationInputs(
            discretization="Milstein",
            sampling="Quasi",
            num_paths=32_768,
            num_steps=128,
        )
        result = SimulationManager().run(inputs)

        normal_cdf = lambda value: 0.5 * (
            1.0 + math.erf(value / math.sqrt(2.0))
        )
        benchmark = (
            100.0 * normal_cdf(0.35)
            - 100.0 * math.exp(-0.05) * normal_cdf(0.15)
        )
        self.assertLess(abs(result.option_price - benchmark), 0.35)


if __name__ == "__main__":
    unittest.main()
