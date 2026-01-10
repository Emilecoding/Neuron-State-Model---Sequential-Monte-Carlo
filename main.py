from generate_data import generate_observations, simulate_hh, plot_observations
from particle_filter import particle_filter, plot_signal
from smoothing import (
    backward_smoothing_logdomain,
    smooth_expectation,
    plot_smoothing_results,
)
from em_algorithm import em_algorithm_hh, plot_em_convergence
import matplotlib.pyplot as plt
import numpy as np

"""

---------------------------- Generate data ----------------------------

"""

T = 4000
dt = 0.01
I = 10
sigma_dyn = 0.1
sigma_obs = 1
Ds = 5
traj_init = [-65.0, 0.05, 0.6, 0.32]

traj = simulate_hh(
    T=T, dt=dt, I=I, noise_std=sigma_dyn, traj_init=traj_init
)  # (V, m, h, n)
V_true = traj[:, 0]

V_obs = generate_observations(V=V_true, sigma_obs=sigma_obs, Ds=Ds)
# plot_observations(V_obs, V_true)

print("---------------Generating data is finished---------------")
"""

---------------------------- Particle filter ----------------------------

"""

N_particles = 200

V_filt, particles, weights = particle_filter(
    y=V_obs,
    N=N_particles,
    dt=dt,
    I=I,
    sigma_dyn=sigma_dyn,
    sigma_obs=sigma_obs,
    save_particles=True,
    particle_init=traj_init,
)

# plot_signal(V_filt, V_obs, V_true, dt)
print("---------------Particle filter is finished---------------")

"""

---------------------------- Smoothing ----------------------------

"""
smoothed_particles, smoothed_weights = backward_smoothing_logdomain(
    particles=particles, weights=weights, dt=dt, I=I, sigma_dyn=sigma_dyn
)

expectations = smooth_expectation(smoothed_particles, smoothed_weights)
V_smooth = expectations[:, 0]

# plot_smoothing_results(V_true, V_filt, V_smooth, V_obs, dt)

print("---------------Smoothing is finished---------------")

"""

---------------------------- EM algorithm ----------------------------

"""


initial_params = {
    "sigma_obs": 2,
    "sigma_dyn": 0.3,
    "I": 10.0,
    "gNa": 100.0,  # Intentionnellement différent des vraies valeurs
    "gK": 30.0,
    "gL": 0.35,
    "ENa": 55.0,
    "EK": -70.0,
    "EL": -54.4,
}
true_params = {
    "gNa": 120.0,
    "gK": 36.0,
    "gL": 0.3,
    "ENa": 50.0,
    "EK": -77.0,
    "EL": -54.4,
    "sigma_obs": 1.0,
    "sigma_dyn": 0.1,
}

n_iter = 3
params_final, history = em_algorithm_hh(
    y_obs=V_obs[:1000],
    dt=dt,
    n_iter=n_iter,
    N_particles=100,
    initial_params=initial_params,
)

plot_em_convergence(history=history, true_params=true_params)
