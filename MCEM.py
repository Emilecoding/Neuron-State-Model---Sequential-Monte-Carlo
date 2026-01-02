import numpy as np
from data import Matrix, Vector, y_obs
import matplotlib.pyplot as plt
from particles import hmm as phmm

def mcem_hmm_gaussian(y: Vector, n_state=2, n_iter=30, n_smooth_paths=200, seed=42, init_params=None) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    T = len(y)

    if init_params is None:
        pi_0 = np.ones(n_state) / n_state
        transition_matrix = np.full((n_state, n_state), 1 / n_state)
        mus = np.quantile(y, np.linspace(0.2, 0.8, n_state))
        sigmas = np.full(n_state, np.std(y) + 1e-6)
    else:
        pi_0 = init_params["pi_0"].copy()
        transition_matrix = init_params["transition_matrix"].copy()
        mus = init_params["mus"].copy()
        sigmas = init_params["sigmas"].copy()

    params_hist = {
        "pi_0": [],
        "transition_matrix": [],
        "mus": [],
        "sigmas": [],
        "loglik": []
    }

    theta = {"pi_0": pi_0, "transition_matrix": transition_matrix, "mus": mus, "sigmas": sigmas}

    for it in range(n_iter):
        model = phmm.GaussianHMM(
            init_dist=pi_0,
            trans_mat=transition_matrix,
            mus=mus,
            sigmas=sigmas
        )

        bw = phmm.BaumWelch(hmm=model, data=y)
        bw.backward()

        loglik = None
        for attr in ["loglik", "logL", "log_like", "llik"]:
            if hasattr(bw, attr):
                loglik = getattr(bw, attr)
                break

        Xs = bw.sample(N=n_smooth_paths)
        Xs = np.asarray(Xs)

        if Xs.ndim != 2:
            raise ValueError("Unexpected format for bw.sample()")

        if Xs.shape == (n_smooth_paths, T):
            paths = Xs
        elif Xs.shape == (T, n_smooth_paths):
            paths = Xs.T
        else:
            if Xs.shape[0] == n_smooth_paths:
                paths = Xs
                T = Xs.shape[1]
            elif Xs.shape[1] == n_smooth_paths:
                paths = Xs.T
                T = paths.shape[1]
            else:
                raise ValueError(f"Unexpected shape for bw.sample(): {Xs.shape}")

        pi0_new = np.bincount(paths[:, 0], minlength=n_state).astype(float)
        pi0_new /= pi0_new.sum()

        trans_counts = np.zeros((n_state, n_state), dtype=float)
        from_counts = np.zeros(n_state, dtype=float)
        for n in range(paths.shape[0]):
            x = paths[n]
            i = x[:-1]
            j = x[1:]
            for a, b in zip(i, j):
                trans_counts[a, b] += 1.0
                from_counts[a] += 1.0

        A_new = np.where(
            from_counts[:, None] > 0,
            trans_counts / from_counts[:, None],
            transition_matrix
        )

        weights = np.zeros((n_state, len(y)), dtype=float)
        for k in range(n_state):
            weights[k] = np.mean(paths == k, axis=0)

        mus_new = np.zeros(n_state)
        sigmas_new = np.zeros(n_state)
        eps = 1e-12
        for k in range(n_state):
            wk = weights[k]
            wsum = wk.sum() + eps
            mus_new[k] = np.sum(wk * y) / wsum
            sigmas_new[k] = np.sqrt(np.sum(wk * (y - mus_new[k]) ** 2) / wsum + 1e-12)

        params_hist["pi_0"].append(pi_0.copy())
        params_hist["transition_matrix"].append(transition_matrix.copy())
        params_hist["mus"].append(mus.copy())
        params_hist["sigmas"].append(sigmas.copy())
        params_hist["loglik"].append(loglik)

        pi_0, transition_matrix, mus, sigmas = pi0_new, A_new, mus_new, sigmas_new
        theta = {"pi_0": pi_0, "transition_matrix": transition_matrix, "mus": mus, "sigmas": sigmas}

    return params_hist, theta


def plot_em_convergence(params_hist, true_params=None):
    mus_hist = np.array(params_hist["mus"])
    sig_hist = np.array(params_hist["sigmas"])

    plt.figure()
    for k in range(mus_hist.shape[1]):
        plt.plot(mus_hist[:, k], label=f"mu[{k}]")
    if true_params is not None:
        for k, v in enumerate(true_params["mus"]):
            plt.axhline(v, linestyle="--", alpha=0.6)
    plt.title("EM Convergence: means (mu)")
    plt.xlabel("EM iteration")
    plt.legend()
    plt.show()

    plt.figure()
    for k in range(sig_hist.shape[1]):
        plt.plot(sig_hist[:, k], label=f"sigma[{k}]")
    if true_params is not None:
        for k, v in enumerate(true_params["sigmas"]):
            plt.axhline(v, linestyle="--", alpha=0.6)
    plt.title("EM convergence: standard deviation (sigma)")
    plt.xlabel("EM iteration")
    plt.legend()
    plt.show()

    A_hist = np.array(params_hist["transition_matrix"])
    plt.figure()
    plt.plot(A_hist[:, 0, 0], label="A00")
    plt.plot(A_hist[:, 0, 1], label="A01")
    plt.plot(A_hist[:, 1, 0], label="A10")
    plt.plot(A_hist[:, 1, 1], label="A11")
    if true_params is not None:
        Atrue = true_params["A"]
        plt.axhline(Atrue[0, 0], linestyle="--", alpha=0.6)
        plt.axhline(Atrue[0, 1], linestyle="--", alpha=0.6)
        plt.axhline(Atrue[1, 0], linestyle="--", alpha=0.6)
        plt.axhline(Atrue[1, 1], linestyle="--", alpha=0.6)
    plt.title("EM Convergence: transitions")
    plt.xlabel("EM iteration")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    transition_matrix_true = np.array([[0.995, 0.005],
                                       [0.01,  0.99]])
    pi0_true = np.array([0.5, 0.5])
    mus_true = np.array([0.0, 1.0])
    sigmas_true = np.array([0.2, 0.3])

    params_hist, theta_hat = mcem_hmm_gaussian(
        y=y_obs,
        n_state=2,
        n_iter=40,
        n_smooth_paths=300,
        seed=123
    )

    print("theta_hat =", theta_hat)

    plot_em_convergence(
        params_hist,
        true_params={"A": transition_matrix_true, "mus": mus_true, "sigmas": sigmas_true}
    )
