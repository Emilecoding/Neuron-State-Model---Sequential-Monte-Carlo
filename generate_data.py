import numpy as np
import matplotlib.pyplot as plt

# =========================
# Hodgkin-Huxley model
# =========================

def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55)/10))
def beta_n(V):  return 0.125 * np.exp(-(V + 65)/80)

def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40)/10))
def beta_m(V):  return 4.0 * np.exp(-(V + 65)/18)

def alpha_h(V): return 0.07 * np.exp(-(V + 65)/20)
def beta_h(V):  return 1 / (1 + np.exp(-(V + 35)/10))


def hh_step(state, I, dt, noise_std=0.0):
    """
    One Euler-Maruyama step of HH dynamics
    state = (V, m, h, n)
    """
    V, m, h, n = state

    # Constants
    C = 1.0
    gNa, gK, gL = 120.0, 36.0, 0.3
    ENa, EK, EL = 50.0, -77.0, -54.4

    INa = gNa * m**3 * h * (V - ENa)
    IK  = gK  * n**4 * (V - EK)
    IL  = gL  * (V - EL)

    dV = (I - INa - IK - IL) / C
    dm = alpha_m(V)*(1-m) - beta_m(V)*m
    dh = alpha_h(V)*(1-h) - beta_h(V)*h
    dn = alpha_n(V)*(1-n) - beta_n(V)*n

    V_new = V + dt*dV + noise_std*np.sqrt(dt)*np.random.randn()
    m_new = m + dt*dm
    h_new = h + dt*dh
    n_new = n + dt*dn

    return np.array([V_new, m_new, h_new, n_new])


def simulate_hh(T=4000, dt=0.01, I=10.0, noise_std=0.0):
    """
    Simulates a Hodgkin-Huxley neuron
    """
    traj = np.zeros((T, 4))
    traj[0] = [-65.0, 0.05, 0.6, 0.32]

    for t in range(1, T):
        traj[t] = hh_step(traj[t-1], I, dt, noise_std)

    return traj


def generate_observations(V, sigma_obs=5.0, Ds=5):
    """
    Observe voltage with noise and subsampling
    """
    y = np.full_like(V, np.nan)
    for t in range(0, len(V), Ds):
        y[t] = V[t] + sigma_obs*np.random.randn()
    return y


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    np.random.seed(0)

    T = 4000
    dt = 0.01

    traj = simulate_hh(T=T, dt=dt, I=10.0, noise_std=0.2)
    V_true = traj[:, 0]

    y_obs = generate_observations(V_true, sigma_obs=5.0, Ds=5)

    np.savez(
        "hh_data.npz",
        traj=traj,
        V_true=V_true,
        y_obs=y_obs,
        dt=dt
    )

    t = np.arange(T)*dt
    plt.figure(figsize=(12,4))
    plt.plot(t, V_true, label="True voltage")
    plt.scatter(t[~np.isnan(y_obs)],
                y_obs[~np.isnan(y_obs)],
                s=10, color="red", label="Observations")
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Hodgkin-Huxley simulation with noisy observations")
    plt.show()