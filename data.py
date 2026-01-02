import numpy as np
import matplotlib.pyplot as plt 
from typing import TypeAlias

Vector : TypeAlias = np.ndarray
Matrix : TypeAlias = np.ndarray

n_state = 2   # open or close
T = 2000    # length of the sequence
transition_matrix = np.array([[0.995, 0.005], [0.01, 0.99]])
pi_0 = np.array([0.5, 0.5]) # initiale distribution
mu = np.array([0.0, 1.0])
sigma = 0.2 

def simulate_markov_chain(transition_matrix: Matrix , pi_0 : Vector, T : int) -> Vector:
    n_state = transition_matrix.shape[0]
    x = np.zeros(T, dtype = int)

    x[0] = np.random.choice(n_state, p = pi_0)

    for t in range(1,T):
        x[t] = np.random.choice(n_state, p = transition_matrix[x[t-1]])

    return x


def generate_observations(x : Vector, mu : Vector, sigma: float) -> Vector:

    T = len(x)
    y = np.zeros(T)

    for t in range(T):
        y[t] = mu[x[t]] + sigma * np.random.randn()
    
    return y

def plot_data(x, y, mu):
    fig, axes = plt.subplots(2,1, figsize = (12,6), sharex = True)

    # Hidden state 
    axes[0].plot(x, lw=1)
    axes[0].set_ylabel("État caché")
    axes[0].set_yticks(range(len(mu)))
    axes[0].set_title("Chaîne de Markov cachée")

    # Observations
    axes[1].plot(y, lw=0.8)
    for m in mu:
        axes[1].axhline(m, linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Observation")
    axes[1].set_xlabel("Temps")
    axes[1].set_title("Observations bruitées")

    plt.tight_layout()
    plt.show()


def main(): 
    np.random.seed(42)

    # Parameters
    n_state = 2   # open or close
    T = 2000    # length of the sequence
    transition_matrix = np.array([[0.995, 0.005], [0.01, 0.99]])
    pi_0 = np.array([0.5, 0.5]) # initiale distribution
    mu = np.array([0.0, 1.0])
    sigma = 0.2 
    
    # Simulation
    x = simulate_markov_chain(transition_matrix, pi_0, T)
    y_obs = generate_observations(x, mu, sigma)
    plot_data(x[:1000], y_obs[:1000], mu)

if __name__ == "__main__":
    main()