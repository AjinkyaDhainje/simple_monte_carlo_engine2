"""Small, reusable time-discretization functions."""

import numpy as np


def euler(current, drift, diffusion, diffusion_derivative, dt, z):
    """Euler-Maruyama: S_next = S + drift*dt + diffusion*sqrt(dt)*Z."""
    del diffusion_derivative  # Euler does not need this value.
    return current + drift * dt + diffusion * np.sqrt(dt) * z


def milstein(current, drift, diffusion, diffusion_derivative, dt, z):
    """Euler plus the Milstein correction for state-dependent diffusion."""
    correction = 0.5 * diffusion * diffusion_derivative * dt * (z**2 - 1.0)
    return current + drift * dt + diffusion * np.sqrt(dt) * z + correction


# The engine uses this registry, so it contains no discretization if-statements.
DISCRETIZATIONS = {
    "Euler": euler,
    "Milstein": milstein,
}

