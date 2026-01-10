import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.special import logsumexp
from generate_data import hh_step
import particles as part
from particles import state_space_models as ssm
from particles import distributions as dists


def backward_smoothing_logdomain(particles, weights, dt, I, sigma_dyn):
    T = len(particles)
    N = particles[0].shape[0]

    if isinstance(particles, np.ndarray):
        particles = [particles[t] for t in range(T)]
    if isinstance(weights, np.ndarray):
        weights = [weights[t] for t in range(T)]

    log_beta = [np.zeros(N) for _ in range(T)]  # log β_T = 0

    for t in range(T - 2, -1, -1):

        # log_transitions[i,j] = log p(x_{t+1}^j | x_t^i)
        log_transitions = np.zeros((N, N))
        for i in range(N):
            pred = hh_step(particles[t][i], I, dt, noise_std=0.0)
            diff = particles[t + 1] - pred.reshape(1, -1)
            log_transitions[i] = -0.5 * np.sum(diff**2, axis=1) / (sigma_dyn**2)

        # log_beta[t][i] = logsumexp_j ( log_transitions[i,j] + log_beta[t+1][j] )
        for i in range(N):
            log_beta[t][i] = logsumexp(log_transitions[i] + log_beta[t + 1])

        # Renormalisation (très important) : enlève une constante
        log_beta[t] -= logsumexp(log_beta[t])

    # Poids lissés en log pour éviter overflow/underflow
    smoothed_weights = []
    for t in range(T):
        log_w = np.log(weights[t] + 1e-300) + log_beta[t]
        log_w -= logsumexp(log_w)
        smoothed_weights.append(np.exp(log_w))

    return np.array(particles), np.array(smoothed_weights)


def smooth_expectation(smoothed_particles, smoothed_weights):
    """
    Compute expectations for all states
    """
    T = len(smoothed_particles)
    state_dim = smoothed_particles[0].shape[1]

    expectations = np.zeros((T, state_dim))

    for t in range(T):
        for d in range(state_dim):
            expectations[t, d] = np.sum(
                smoothed_weights[t] * smoothed_particles[t][:, d]
            )

    return expectations


def plot_smoothing_results(V_true, V_filt, V_smooth, y_obs, dt, save_dir="figures"):
    """Génère et sauvegarde les plots de smoothing"""
    os.makedirs(save_dir, exist_ok=True)

    t = np.arange(len(V_true)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Comparaison complet
    axes[0].plot(t, V_true, "k-", label="Vrai voltage", alpha=0.7, linewidth=1)
    axes[0].plot(t, V_filt, "b-", label="Filtré", alpha=0.5, linewidth=1)
    axes[0].plot(t, V_smooth, "r-", label="Lissé", alpha=0.8, linewidth=1.5)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("Comparaison: Vrai vs Filtré vs Lissé")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. Zoom sur une région
    zoom_start = 1000
    zoom_end = 1500
    mask_zoom = (t >= zoom_start) & (t <= zoom_end)

    axes[1].plot(t[mask_zoom], V_true[mask_zoom], "k-", label="Vrai", alpha=0.7)
    axes[1].plot(t[mask_zoom], V_filt[mask_zoom], "b-", label="Filtré", alpha=0.5)
    axes[1].plot(t[mask_zoom], V_smooth[mask_zoom], "r-", label="Lissé", alpha=0.8)
    axes[1].scatter(
        t[~np.isnan(y_obs) & mask_zoom],
        y_obs[~np.isnan(y_obs) & mask_zoom],
        s=10,
        c="green",
        alpha=0.5,
        label="Observations",
    )
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].set_title(f"Zoom: {zoom_start}-{zoom_end} ms")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 3. Erreurs
    mask_obs = ~np.isnan(y_obs)
    error_filt = np.abs(V_filt[mask_obs] - V_true[mask_obs])
    error_smooth = np.abs(V_smooth[mask_obs] - V_true[mask_obs])

    axes[2].plot(
        t[mask_obs][:300], error_filt[:300], "b-", alpha=0.5, label="Erreur filtré"
    )
    axes[2].plot(
        t[mask_obs][:300], error_smooth[:300], "r-", alpha=0.8, label="Erreur lissé"
    )
    axes[2].set_xlabel("Temps (ms)")
    axes[2].set_ylabel("Erreur absolue (mV)")
    axes[2].set_title(
        f"Erreurs (RMSE filtré: {np.mean(error_filt):.2f} mV, "
        f"lissé: {np.mean(error_smooth):.2f} mV)"
    )
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/smoothing_results.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{save_dir}/smoothing_results.pdf")
    plt.show()

    return fig


def test():
    print("=" * 60)
    print("SMOOTHING BACKWARD - EXÉCUTION PRINCIPALE")
    print("=" * 60)

    # 1. Charger les données ORIGINALES
    print("\n[1/3] Chargement des données originales...")
    data_hh = np.load("hh_data.npz", allow_pickle=True)
    V_true = data_hh["V_true"]
    y_obs = data_hh["y_obs"]
    dt = data_hh["dt"].item()

    print(f"   ✓ Données HH chargées: {len(V_true)} points, dt={dt}")

    # 2. Charger les résultats du particle filter
    print("\n[2/3] Chargement des résultats du particle filter...")
    try:
        data_pf = np.load("results/particle_filter_results.npz", allow_pickle=True)

        particles = data_pf["particles"]
        weights = data_pf["weights"]
        V_filt = data_pf["V_filt"]
        I = data_pf["I"].item()
        sigma_dyn = data_pf["sigma_dyn"].item()

        print(f"   ✓ Particle filter chargé:")
        print(f"     - particles shape: {particles.shape}")
        print(f"     - weights shape: {weights.shape}")
        print(f"     - I = {I}, sigma_dyn = {sigma_dyn}")

    except FileNotFoundError:
        print("   ❌ Fichier non trouvé: results/particle_filter_results.npz")
        print("   Exécutez d'abord: python particle_filter.py")
        exit(1)
    except KeyError as e:
        print(f"   ❌ Clé manquante: {e}")
        print("   Le fichier .npz ne contient pas les données attendues")
        exit(1)

    # 3. Appliquer le smoothing
    print("\n[3/3] Application du smoothing backward...")

    # Utiliser seulement une partie pour tester plus rapidement
    TEST_SIZE = 1000  # Réduire pour tests rapides
    if len(particles) > TEST_SIZE:
        print(f"   ⚠  Utilisation des {TEST_SIZE} premiers points pour test rapide")
        particles = particles[:TEST_SIZE]
        weights = weights[:TEST_SIZE]
        V_true = V_true[:TEST_SIZE]
        V_filt = V_filt[:TEST_SIZE]
        y_obs = y_obs[:TEST_SIZE]

    # Exécuter le smoothing

    start_time = time.time()

    smoothed_particles, smoothed_weights = backward_smoothing_logdomain(
        particles, weights, dt, I, sigma_dyn
    )

    elapsed = time.time() - start_time
    print(f"   ✓ Smoothing terminé en {elapsed:.2f} secondes")

    # Convertir en arrays numpy pour facilité
    smoothed_particles_array = np.array(smoothed_particles)
    smoothed_weights_array = np.array(smoothed_weights)

    # 4. Calculer les attentes
    print("\n[4/4] Calcul des estimations lissées...")
    expectations = smooth_expectation(smoothed_particles_array, smoothed_weights_array)
    V_smooth = expectations[:, 0]  # Voltage seulement

    # 5. Sauvegarder les résultats
    print("\n💾 Sauvegarde des résultats...")
    os.makedirs("results", exist_ok=True)

    np.savez(
        "results/smoothed_results.npz",
        particles=smoothed_particles_array,
        weights=smoothed_weights_array,
        V_smooth=V_smooth,
        V_filt=V_filt,
        V_true=V_true,
        y_obs=y_obs,
        dt=dt,
        I=I,
        sigma_dyn=sigma_dyn,
        expectations=expectations,
    )

    print("Résultats sauvegardés dans: results/smoothed_results.npz")
    print(f"     - smoothed_particles: {smoothed_particles_array.shape}")
    print(f"     - smoothed_weights: {smoothed_weights_array.shape}")

    # 6. Calculer les erreurs
    mask = ~np.isnan(y_obs)
    rmse_filt = np.sqrt(np.mean((V_filt[mask] - V_true[mask]) ** 2))
    rmse_smooth = np.sqrt(np.mean((V_smooth[mask] - V_true[mask]) ** 2))

    print(f"PERFORMANCE:")
    print(f"   RMSE filtré:  {rmse_filt:.3f} mV")
    print(f"   RMSE lissé:   {rmse_smooth:.3f} mV")
    print(f"   Amélioration: {100*(rmse_filt - rmse_smooth)/rmse_filt:.1f}%")

    # 7. Générer les plots
    print("Génération des graphiques...")
    fig = plot_smoothing_results(V_true, V_filt, V_smooth, y_obs, dt)

    print("\n" + "=" * 60)
    print("SMOOTHING TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)


if __name__ == "__main__":
    test()
