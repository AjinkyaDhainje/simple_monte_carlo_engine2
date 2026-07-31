# Simple Configurable Monte Carlo Engine

This project prices call and put options while keeping the code deliberately
small and readable. It includes:

- Geometric and Arithmetic Brownian Motion models
- One-factor and two-factor Hull-White models
- Merton jump-diffusion and Variance-Gamma Levy-process models
- Euler and Milstein discretizations
- Vanilla and arithmetic-average Asian payoffs
- Standard, Sobol quasi-Monte Carlo, and scrambled Sobol sampling
- Antithetic variance reduction for generated paths
- Up to 1,000,000 simulated paths
- Repeated runs with every final option price and their average
- Option price, standard error, and approximate 95% confidence interval
- Path, terminal-price, payoff, and price-convergence charts
- A Streamlit UI with model/payoff inputs that appear immediately when needed

There is no user-facing random seed.

## Project structure

```text
simple_monte_carlo_engine/
├── app.py
├── monte_carlo/
│   ├── inputs.py             # One object containing every possible input
│   ├── discretizations.py    # Euler and Milstein functions
│   ├── sampling.py           # Standard and Sobol Z generators
│   ├── engine.py             # Model equations and batched path generation
│   ├── payoffs.py            # Payoff classes and payoff registry
│   ├── manager.py            # Main pricing flow and result
│   └── charts.py             # UI charts
└── tests/test_engine.py
```

The central flow is intentionally short:

```text
inputs -> manager -> merged model/engine -> selected payoff -> result
```

## Run it

Python 3.10 or newer is recommended.

```bash
cd simple_monte_carlo_engine
python -m venv .venv
```

Activate the environment:

```bash
# macOS or Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with:

```bash
python -m unittest discover -v
```

## How the inputs stay simple

`SimulationInputs` contains common inputs and all model/payoff-specific inputs.
There are no separate Merton-input or Asian-input classes. The selected model
or payoff reads only what it needs:

- Every model ignores optional fields belonging to the other models.
- Vanilla ignores `averaging_days`.
- Merton validates and reads the jump fields.
- Hull-White reads its mean-reversion fields only.
- The Levy process reads `levy_skew` and `levy_variance` only.
- Asian validates and reads `averaging_days`.

The model and payoff classes declare which extra input fields the UI should
show. Streamlit reruns as soon as a selection changes, so these fields appear
directly below the selected component.

## Model equations

- **Geometric Brownian Motion:** `dS = r*S*dt + sigma*S*dW`.
- **Arithmetic Brownian Motion:** `dS = r*S0*dt + sigma*S0*dW`.
- **Hull-White:** `dr = a*(b-r)*dt + sigma*dW`.
- **2f Hull White:** two correlated mean-reverting factors, with
  `r = long_term_mean + x + y`.
- **Merton Jump Model:** risk-neutral GBM plus lognormal compound-Poisson jumps.
- **Levy Process:** an exponential Variance-Gamma process. Its increments are
  sampled directly, so Euler and Milstein produce the same transition.

The source contains comments beside each implementation explaining the full
equation and the meaning of its parameters.

## Large simulations and UI data

The manager uses a normal `for` loop and asks the engine for one path set per
batch. With 1,000,000 requested paths:

- all 1,000,000 paths contribute to the payoff calculation;
- all 1,000,000 discounted payoffs contribute to the option price, standard
  error, confidence interval, and convergence calculation;
- all 1,000,000 terminal prices are available to the final-price chart;
- only the first 10,000 complete paths are retained for the UI;
- 100 evenly spaced paths from that 10,000-path subset are plotted.

The 10,000-path limit affects only path visualisation, never pricing.
Within each batch, the engine generates half of the normal draws and mirrors
them as `Z, -Z` antithetic pairs.

When multiple runs are requested, the UI displays each run's final option
price and their average. Full path and payoff data is retained only for the
final run, and all four charts are shown below the complete price summary.

## Add another model

1. Add any new input fields to `SimulationInputs` with sensible defaults.
2. Add one class in `engine.py`, usually by extending `DiffusionModel`.
3. Put UI-only field descriptions in that class's `extra_inputs` list.
4. Add one entry to the `MODELS` registry above `MonteCarloEngine`.

The engine, manager, and UI do not need a new model-specific `if` statement.

## Add another payoff

1. Add any new input fields to `SimulationInputs` with sensible defaults.
2. Add one small class in `payoffs.py` with `validate` and `calculate` methods.
3. Put UI-only field descriptions in its `extra_inputs` list.
4. Add one entry to the `PAYOFFS` registry at the bottom of `payoffs.py`.

The manager and UI do not need a new payoff-specific `if` statement.

## Numerical notes

- Rates and volatility are decimals, so `0.05` means 5%.
- The Asian input is in days and averages the final simulated observations.
- The reported confidence interval is the estimate plus or minus 1.96 standard
  errors.
- Sobol sequences work with any positive path count, but powers of two retain
  their strongest balance properties.
- Euler and Milstein are educational approximations. GBM has an exact
  transition, but this project intentionally compares the requested numerical
  discretizations.
