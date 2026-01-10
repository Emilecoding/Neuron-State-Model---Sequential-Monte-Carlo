import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from generate_data import simulate_hh, generate_observations
from particle_filter import particle_filter
from smoothing import (
    backward_smoothing_corrected,
    smooth_expectation,
    backward_smoothing_logdomain,
)
from maximisation import compute_sigma_obs, compute_sigma_dyn, estimate_hh_parameters


def em_algorithm_hh(
    y_obs,
    dt,
    n_iter=5,
    N_particles=200,
    initial_params=None,
    true_values=[120.0, 36.0, 0.3, 50.0, -77.0, -54.4],
):
    """
    Algorithme EM complet pour le modèle HH

    Parameters
    ----------
    y_obs : array
        Observations bruitées
    dt : float
        Pas de temps
    n_iter : int
        Nombre d'itérations EM
    N_particles : int
        Nombre de particules pour le filtrage
    initial_params : dict
        Paramètres initiaux
    """

    T = len(y_obs)

    # Initialisation des paramètres
    if initial_params is None:
        params = {
            "sigma_obs": 5.0,
            "sigma_dyn": 0.3,
            "I": 10.0,
            "gNa": 100.0,  # Intentionnellement différent des vraies valeurs
            "gK": 30.0,
            "gL": 0.3,
            "ENa": 50.0,
            "EK": -77.0,
            "EL": -54.4,
        }
    else:
        params = initial_params.copy()

    # Historique pour tracking
    history = {
        "params": [],
        "rmse_filt": [],
        "rmse_smooth": [],
        "V_smooth": [],
        "time": [],
    }

    print("=" * 70)
    print("ALGORITHME EM COMPLET POUR HODGKIN-HUXLEY")
    print(f"Itérations: {n_iter}, Particules: {N_particles}")
    print("=" * 70)

    for iteration in range(n_iter):
        print(f"\n{'='*70}")
        print(f"ITERATION EM {iteration+1}/{n_iter}")
        print("=" * 70)

        iter_start = time.time()

        # ===========================================
        # E-STEP: Filtrage et Smoothing
        # ===========================================
        print("\n🎯 E-STEP: Filtrage particulaire...")

        # 1. Filtrage
        start = time.time()
        V_filt, particles, weights = particle_filter(
            y_obs,
            N=N_particles,
            dt=dt,
            I=params["I"],
            sigma_dyn=params["sigma_dyn"],
            sigma_obs=params["sigma_obs"],
            save_particles=True,
        )
        filter_time = time.time() - start
        print(f"   ✓ Filtrage terminé ({filter_time:.1f}s)")

        # 2. Smoothing
        print("   Smoothing backward...")
        start = time.time()

        # Convertir en listes pour le smoothing
        particles_list = [particles[t] for t in range(T)]
        weights_list = [weights[t] for t in range(T)]

        smoothed_particles_list, smoothed_weights_list = backward_smoothing_logdomain(
            particles_list, weights_list, dt, params["I"], params["sigma_dyn"]
        )

        # Convertir en arrays
        smoothed_particles = np.array(smoothed_particles_list)
        smoothed_weights = np.array(smoothed_weights_list)

        smooth_time = time.time() - start
        print(f"   ✓ Smoothing terminé ({smooth_time:.1f}s)")

        # Calculer l'estimation lissée
        V_smooth = np.sum(smoothed_weights * smoothed_particles[:, :, 0], axis=1)

        # ===========================================
        # M-STEP: Estimation des paramètres
        # ===========================================
        print("\n🎯 M-STEP: Estimation des paramètres...")

        # 1. Estimation des bruits
        print("   Estimation des variances de bruit...")
        sigma_obs_new = compute_sigma_obs(smoothed_particles, y_obs, smoothed_weights)
        sigma_dyn_new = compute_sigma_dyn(
            smoothed_particles, smoothed_weights, dt, params["I"]
        )

        print(
            f"   ✓ sigma_obs: {sigma_obs_new:.3f} (précédent: {params['sigma_obs']:.3f})"
        )
        print(
            f"   ✓ sigma_dyn: {sigma_dyn_new:.3f} (précédent: {params['sigma_dyn']:.3f})"
        )

        # 2. Estimation des paramètres HH
        print("   Estimation des paramètres HH...")
        start = time.time()

        # Paramètres à estimer
        hh_params = estimate_hh_parameters(
            smoothed_particles, smoothed_weights, dt, params["I"], sigma_dyn_new
        )

        gNa_new, gK_new, gL_new, ENa_new, EK_new, EL_new = hh_params

        hh_time = time.time() - start
        print(f"   ✓ Estimation HH terminée ({hh_time:.1f}s)")

        # ===========================================
        # Mise à jour des paramètres
        # ===========================================
        old_params = params.copy()

        params.update(
            {
                "sigma_obs": sigma_obs_new,
                "sigma_dyn": sigma_dyn_new,
                "gNa": gNa_new,
                "gK": gK_new,
                "gL": gL_new,
                "ENa": ENa_new,
                "EK": EK_new,
                "EL": EL_new,
            }
        )

        # ===========================================
        # Calcul des métriques
        # ===========================================
        # Charger les vraies valeurs pour comparaison
        try:
            data_true = np.load("hh_data.npz", allow_pickle=True)
            V_true = data_true["V_true"]

            mask = ~np.isnan(y_obs)
            rmse_filt = np.sqrt(np.mean((V_filt[mask] - V_true[mask]) ** 2))
            rmse_smooth = np.sqrt(np.mean((V_smooth[mask] - V_true[mask]) ** 2))

            print(f"\n📊 Métriques (itération {iteration+1}):")
            print(f"   RMSE filtré:  {rmse_filt:.3f} mV")
            print(f"   RMSE lissé:   {rmse_smooth:.3f} mV")

        except:
            rmse_filt = rmse_smooth = np.nan
            print("   ⚠  Impossible de calculer RMSE (fichier hh_data.npz manquant)")

        # ===========================================
        # Sauvegarde des résultats de l'itération
        # ===========================================
        iter_time = time.time() - iter_start

        history["params"].append(params.copy())
        history["rmse_filt"].append(rmse_filt)
        history["rmse_smooth"].append(rmse_smooth)
        history["V_smooth"].append(V_smooth.copy())
        history["time"].append(iter_time)

        print(f"\n⏱️  Temps itération: {iter_time:.1f}s")

        # Afficher l'évolution des paramètres
        print("\n📈 Évolution des paramètres HH:")
        param_names = ["gNa", "gK", "gL", "ENa", "EK", "EL"]

        for name, true in zip(param_names, true_values):
            old = old_params[name]
            new = params[name]
            error_old = 100 * abs(old - true) / true
            error_new = 100 * abs(new - true) / true

            arrow = "→" if abs(new - true) < abs(old - true) else "←"
            print(
                f"   {name}: {old:6.2f} → {new:6.2f} (vrai: {true:6.2f}) "
                f"{arrow} erreur: {error_old:5.1f}% → {error_new:5.1f}%"
            )

        # Sauvegarder les résultats intermédiaires
        os.makedirs("results/em_iterations", exist_ok=True)

        np.savez(
            f"results/em_iterations/iteration_{iteration+1:02d}.npz",
            V_filt=V_filt,
            V_smooth=V_smooth,
            particles=particles,
            weights=weights,
            smoothed_particles=smoothed_particles,
            smoothed_weights=smoothed_weights,
            **params,
        )

        print(
            f"💾 Résultats sauvegardés: results/em_iterations/iteration_{iteration+1:02d}.npz"
        )

    # ===========================================
    # Final: Sauvegarde globale et visualisation
    # ===========================================
    print(f"\n{'='*70}")
    print("ALGORITHME EM TERMINÉ")
    print("=" * 70)

    # Sauvegarder l'historique complet
    final_results = {
        "final_params": params,
        "history": history,
        "n_iter": n_iter,
        "N_particles": N_particles,
        "dt": dt,
        "T": T,
    }

    np.savez("results/em_final_results.npz", **final_results)

    # Sauvegarder aussi en JSON pour lisibilité
    json_params = params.copy()
    # Convertir les numpy arrays en listes pour JSON
    for key in json_params:
        if isinstance(json_params[key], np.ndarray):
            json_params[key] = json_params[key].tolist()
        elif isinstance(json_params[key], np.generic):
            json_params[key] = json_params[key].item()

    with open("results/em_final_params.json", "w") as f:
        json.dump(json_params, f, indent=2)

    print("📁 Résultats sauvegardés:")
    print("   - results/em_final_results.npz")
    print("   - results/em_final_params.json")
    print("   - results/em_iterations/iteration_*.npz")

    return params, history


def plot_em_convergence(history, true_params=None):
    """
    Visualise la convergence de l'algorithme EM
    """
    os.makedirs("figures/em", exist_ok=True)

    n_iter = len(history["params"])
    iterations = range(1, n_iter + 1)

    # 1. Convergence des paramètres HH
    param_names = true_params.keys()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, ax) in enumerate(zip(param_names, axes)):
        values = [p[name] for p in history["params"]]

        ax.plot(iterations, values, "o-", linewidth=2, markersize=8)

        if true_params and name in true_params:
            ax.axhline(
                y=true_params[name],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Valeur vraie",
            )

        ax.set_xlabel("Itération EM")
        ax.set_ylabel(name)
        ax.set_title(f"Convergence de {name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("figures/em/parameter_convergence.png", dpi=150, bbox_inches="tight")

    # 2. Convergence des bruits
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    sigma_obs_vals = [p["sigma_obs"] for p in history["params"]]
    sigma_dyn_vals = [p["sigma_dyn"] for p in history["params"]]

    axes2[0].plot(iterations, sigma_obs_vals, "o-", linewidth=2)
    axes2[0].axhline(
        y=true_params["sigma_obs"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Vrai {true_params["sigma_obs"]}",
    )
    axes2[0].set_xlabel("Itération EM")
    axes2[0].set_ylabel("sigma_obs")
    axes2[0].set_title("Convergence du bruit observation")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    axes2[1].plot(iterations, sigma_dyn_vals, "o-", linewidth=2, color="orange")
    axes2[1].axhline(
        y=true_params["sigma_dyn"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Vrai {true_params["sigma_dyn"]}",
    )
    axes2[1].set_xlabel("Itération EM")
    axes2[1].set_ylabel("sigma_dyn")
    axes2[1].set_title("Convergence du bruit dynamique")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig("figures/em/noise_convergence.png", dpi=150, bbox_inches="tight")

    # 3. Convergence du RMSE
    if not np.all(np.isnan(history["rmse_smooth"])):
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        ax3.plot(
            iterations,
            history["rmse_filt"],
            "o-",
            linewidth=2,
            label="RMSE filtré",
            alpha=0.7,
        )
        ax3.plot(
            iterations,
            history["rmse_smooth"],
            "s-",
            linewidth=2,
            label="RMSE lissé",
            alpha=0.9,
        )

        ax3.set_xlabel("Itération EM")
        ax3.set_ylabel("RMSE (mV)")
        ax3.set_title("Évolution de l'erreur d'estimation")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig("figures/em/rmse_convergence.png", dpi=150, bbox_inches="tight")

    # 4. Comparaison finale des estimations
    if n_iter >= 2:
        fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))

        # Charger les vraies données
        try:
            data_true = np.load("hh_data.npz", allow_pickle=True)
            V_true = data_true["V_true"]
            y_obs = data_true["y_obs"]

            # Première itération
            axes4[0, 0].plot(V_true[:500], "k-", alpha=0.7, label="Vrai")
            axes4[0, 0].plot(
                history["V_smooth"][0][:500], "b-", alpha=0.5, label="Itération 1"
            )
            axes4[0, 0].set_title("Estimation - Itération 1")
            axes4[0, 0].set_xlabel("Time step")
            axes4[0, 0].set_ylabel("Voltage (mV)")
            axes4[0, 0].legend()
            axes4[0, 0].grid(True, alpha=0.3)

            # Dernière itération
            axes4[0, 1].plot(V_true[:500], "k-", alpha=0.7, label="Vrai")
            axes4[0, 1].plot(
                history["V_smooth"][-1][:500],
                "r-",
                alpha=0.8,
                label=f"Itération {n_iter}",
            )
            axes4[0, 1].set_title(f"Estimation - Itération {n_iter}")
            axes4[0, 1].set_xlabel("Time step")
            axes4[0, 1].set_ylabel("Voltage (mV)")
            axes4[0, 1].legend()
            axes4[0, 1].grid(True, alpha=0.3)

            # Erreur
            axes4[1, 0].plot(
                np.abs(history["V_smooth"][0][:200] - V_true[:200]),
                "b-",
                alpha=0.5,
                label="Itération 1",
            )
            axes4[1, 0].plot(
                np.abs(history["V_smooth"][-1][:200] - V_true[:200]),
                "r-",
                alpha=0.8,
                label=f"Itération {n_iter}",
            )
            axes4[1, 0].set_title("Erreur d'estimation")
            axes4[1, 0].set_xlabel("Time step")
            axes4[1, 0].set_ylabel("Erreur absolue (mV)")
            axes4[1, 0].legend()
            axes4[1, 0].grid(True, alpha=0.3)

            # Amélioration
            improvement = 100 * (
                1 - history["rmse_smooth"][-1] / history["rmse_smooth"][0]
            )
            axes4[1, 1].text(
                0.1,
                0.7,
                f'RMSE initial: {history["rmse_smooth"][0]:.3f} mV',
                fontsize=12,
            )
            axes4[1, 1].text(
                0.1,
                0.5,
                f'RMSE final: {history["rmse_smooth"][-1]:.3f} mV',
                fontsize=12,
            )
            axes4[1, 1].text(
                0.1,
                0.3,
                f"Amélioration: {improvement:.1f}%",
                fontsize=14,
                fontweight="bold",
            )
            axes4[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig(
                "figures/em/estimation_comparison.png", dpi=150, bbox_inches="tight"
            )

        except:
            print(
                "⚠  Impossible de créer les comparaisons (fichier hh_data.npz manquant)"
            )

    plt.show()

    print("📈 Graphiques sauvegardés dans figures/em/")

    return fig


def run_em_from_scratch():
    """
    Exécute l'algorithme EM depuis le début
    """
    print("🚀 LANCEMENT DE L'ALGORITHME EM COMPLET")
    print("=" * 70)

    # Vérifier/Créer les dossiers
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures/em", exist_ok=True)

    # Vérifier que les données existent
    if not os.path.exists("hh_data.npz"):
        print("❌ Fichier hh_data.npz manquant")
        print("Exécutez d'abord: python generate_data.py")
        return

    # Charger les données
    data = np.load("hh_data.npz", allow_pickle=True)
    y_obs = data["y_obs"]
    dt = data["dt"].item()

    print(f"📊 Données chargées:")
    print(f"   - Taille: {len(y_obs)} points")
    print(f"   - dt: {dt} ms")
    print(f"   - Observations: {np.sum(~np.isnan(y_obs))} points")

    # Exécuter EM avec peu d'itérations pour test
    print(f"\n🎯 Démarrage de l'algorithme EM...")
    print("   (Utilisation de paramètres réduits pour test rapide)")

    # Paramètres pour test rapide
    n_iter = 5  # Seulement 3 itérations pour test
    N_particles = 200  # Peu de particules pour vitesse

    params_final, history = em_algorithm_hh(
        y_obs[:1000],  # Premiers 1000 points seulement
        dt,
        n_iter=n_iter,
        N_particles=N_particles,
    )

    # Générer les visualisations
    print("\n📈 Génération des graphiques de convergence...")

    true_params = {
        "gNa": 120.0,
        "gK": 36.0,
        "gL": 0.3,
        "ENa": 50.0,
        "EK": -77.0,
        "EL": -54.4,
        "sigma_obs": 5.0,
        "sigma_dyn": 0.3,
    }

    plot_em_convergence(history, true_params)

    print(f"\n{'='*70}")
    print("✅ ALGORITHME EM TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)

    # Afficher le résumé final
    print("\n📋 RÉSUMÉ FINAL:")
    print("-" * 50)

    param_names = ["gNa", "gK", "gL", "ENa", "EK", "EL"]
    for name in param_names:
        estimated = params_final[name]
        true = true_params[name]
        error_pct = 100 * abs(estimated - true) / true

        arrow = "✓" if error_pct < 20 else "⚠"
        print(
            f"{arrow} {name}: {estimated:6.2f} (vrai: {true:6.2f}) | erreur: {error_pct:5.1f}%"
        )

    print(
        f"\n🎯 Amélioration RMSE: {100*(1-history['rmse_smooth'][-1]/history['rmse_smooth'][0]):.1f}%"
    )
    print(f"⏱️  Temps total: {sum(history['time']):.1f}s")

    return params_final, history


if __name__ == "__main__":
    run_em_from_scratch()
