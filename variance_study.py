import os
import json
import numpy as np
import matplotlib.pyplot as plt

from em_algorithm import em_algorithm_hh


def _extract_param_series(history, key):
    """Retourne la série [p[key] for p in history['params']] en np.array."""
    return np.array([p[key] for p in history["params"]], dtype=float)


def run_em_multiple(
    y_obs,
    dt,
    n_runs=10,
    n_iter=10,
    N_particles=200,
    base_seed=0,
    init_mode="same",  # "same" or "random"
    init_spread=0.2,
    save_dir="results/em_multi",
):
    """
    Lance ton EM n_runs fois.

    init_mode:
      - "same": même initialisation à chaque run (variance due au SMC uniquement)
      - "random": initialisation perturbée (variance due aussi aux optima locaux)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialisation "référence" (copie de ton code)
    base_init = {
        "sigma_obs": 1.6,
        "sigma_dyn": 0.3,
        "I": 10.0,
        "gNa": 100.0,
        "gK": 30.0,
        "gL": 0.3,
        "ENa": 50.0,
        "EK": -77.0,
        "EL": -54.4,
    }

    param_keys = ["sigma_obs", "sigma_dyn", "gNa", "gK", "gL", "ENa", "EK", "EL"]

    runs = []
    for r in range(n_runs):
        seed = base_seed + r

        if init_mode == "same":
            init_params = base_init.copy()

        elif init_mode == "random":
            init_params = base_init.copy()
            # perturbations multiplicatives pour gNa,gK,gL et additives pour E*
            for k in ["gNa", "gK", "gL"]:
                init_params[k] = float(
                    init_params[k] * (1.0 + init_spread * np.random.randn())
                )
            for k in ["ENa", "EK", "EL"]:
                init_params[k] = float(
                    init_params[k] + 5.0 * init_spread * np.random.randn()
                )
            for k in ["sigma_obs", "sigma_dyn"]:
                init_params[k] = float(
                    max(1e-3, init_params[k] * (1.0 + init_spread * np.random.randn()))
                )
        else:
            raise ValueError("init_mode must be 'same' or 'random'.")

        print(f"\n=== RUN {r+1}/{n_runs} | seed={seed} | init_mode={init_mode} ===")
        final_params, history = em_algorithm_hh(
            y_obs=y_obs,
            dt=dt,
            n_iter=n_iter,
            N_particles=N_particles,
            initial_params=init_params,
            seed=seed,  # nécessite la petite modif conseillée
        )

        run_obj = {
            "seed": seed,
            "init_params": init_params,
            "final_params": final_params,
            "history": {
                "params": history["params"],
                "rmse_filt": history["rmse_filt"],
                "rmse_smooth": history["rmse_smooth"],
                "time": history["time"],
            },
        }
        runs.append(run_obj)

        # sauvegarde individuelle
        with open(os.path.join(save_dir, f"run_{r+1:02d}.json"), "w") as f:
            json.dump(run_obj, f, indent=2)

    # Agrégation : matrices (n_runs, n_iter)
    series = {k: np.zeros((n_runs, n_iter), dtype=float) for k in param_keys}
    rmse_smooth = np.full((n_runs, n_iter), np.nan)

    for r in range(n_runs):
        hist = runs[r]["history"]
        rmse_smooth[r, :] = np.array(hist["rmse_smooth"], dtype=float)
        for k in param_keys:
            vals = [p[k] for p in hist["params"]]
            series[k][r, :] = np.array(vals, dtype=float)

    summary = {
        "n_runs": n_runs,
        "n_iter": n_iter,
        "N_particles": N_particles,
        "init_mode": init_mode,
        "base_seed": base_seed,
        "final_params": {
            k: [runs[r]["final_params"][k] for r in range(n_runs)] for k in param_keys
        },
        "param_keys": param_keys,
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return runs, series, rmse_smooth


def plot_multi_convergence(
    series, rmse_smooth=None, true_params=None, save_dir="figures/em_multi"
):
    """
    Trace:
      - trajectoires individuelles (run-wise) + moyenne ± écart-type
      - distribution des paramètres finaux
    """
    os.makedirs(save_dir, exist_ok=True)
    n_runs, n_iter = next(iter(series.values())).shape
    it = np.arange(1, n_iter + 1)

    # --- 1) Trajectoires de convergence par paramètre
    for k, mat in series.items():
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        plt.figure(figsize=(9, 5))
        for r in range(n_runs):
            plt.plot(it, mat[r, :], alpha=0.35)
        plt.plot(it, mean, linewidth=3)
        plt.fill_between(it, mean - std, mean + std, alpha=0.2)

        if true_params is not None and k in true_params:
            plt.axhline(true_params[k], linestyle="--")

        plt.xlabel("Itération EM")
        plt.ylabel(k)
        plt.title(f"Convergence multi-runs : {k}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"convergence_{k}.png"), dpi=150)
        plt.close()

    # --- 2) RMSE (si dispo)
    if rmse_smooth is not None and not np.all(np.isnan(rmse_smooth)):
        mean = np.nanmean(rmse_smooth, axis=0)
        std = np.nanstd(rmse_smooth, axis=0)

        plt.figure(figsize=(9, 5))
        for r in range(rmse_smooth.shape[0]):
            plt.plot(it, rmse_smooth[r, :], alpha=0.35)
        plt.plot(it, mean, linewidth=3)
        plt.fill_between(it, mean - std, mean + std, alpha=0.2)

        plt.xlabel("Itération EM")
        plt.ylabel("RMSE lissé (mV)")
        plt.title("Convergence multi-runs : RMSE lissé")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "convergence_rmse_smooth.png"), dpi=150)
        plt.close()

    # --- 3) Histogrammes des paramètres finaux
    for k, mat in series.items():
        finals = mat[:, -1]
        plt.figure(figsize=(7, 4))
        plt.hist(finals, bins=10)
        if true_params is not None and k in true_params:
            plt.axvline(true_params[k], linestyle="--")
        plt.xlabel(f"{k} final")
        plt.ylabel("count")
        plt.title(f"Distribution des optima finaux : {k}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"final_hist_{k}.png"), dpi=150)
        plt.close()

    print(f"Figures sauvegardées dans {save_dir}/")
