import numpy as np
import matplotlib.pyplot as plt
from generate_data import hh_step
import time
import os


def particle_filter(
    y,
    N=200,
    dt=0.01,
    I=10.0,
    sigma_dyn=0.3,
    sigma_obs=5.0,
    save_particles=False,
    particle_init=[-65.0, 0.05, 0.6, 0.32],
):
    """
    Particle filter for HH voltage only (full state propagated)

    Parameters:
    -----------
    save_particles : bool
        Si True, sauvegarde toutes les particules et poids
    """
    T = len(y)

    # Particles: (V, m, h, n)
    particles = np.zeros((N, 4))
    particles[:, 0] = particle_init[0] + 5 * np.random.randn(N)
    particles[:, 1:] = particle_init[1:]

    weights = np.ones(N) / N
    V_est = np.zeros(T)

    # Stockage des particules et poids si demandé
    if save_particles:
        all_particles = np.zeros((T, N, 4))
        all_weights = np.zeros((T, N))

    for t in range(T):
        # Propagation
        for i in range(N):
            particles[i] = hh_step(particles[i], I, dt, noise_std=sigma_dyn)

        # Weight update
        if not np.isnan(y[t]):
            likelihoods = np.exp(-0.5 * ((y[t] - particles[:, 0]) ** 2) / sigma_obs**2)
            weights *= likelihoods + 1e-12
            weights /= np.sum(weights)

        # Effective sample size
        Neff = 1.0 / np.sum(weights**2)

        # Resampling
        if Neff < N / 2:
            idx = np.random.choice(N, size=N, p=weights)
            particles = particles[idx]
            weights[:] = 1.0 / N

        V_est[t] = np.sum(weights * particles[:, 0])

        # Stockage
        if save_particles:
            all_particles[t] = particles.copy()
            all_weights[t] = weights.copy()

    if save_particles:
        return V_est, all_particles, all_weights
    else:
        return V_est


def plot_signal(V_true, V_obs, V_filt, dt):
    mask = ~np.isnan(V_obs)
    rmse = np.sqrt(np.mean((V_filt[mask] - V_true[mask]) ** 2))

    t = np.arange(len(V_true)) * dt
    plt.figure(figsize=(12, 4))
    plt.plot(t, V_true, label="True voltage", lw=2)
    plt.plot(t, V_filt, label="Filtered estimate", lw=2, alpha=0.8)
    plt.scatter(
        t[~np.isnan(V_obs)],
        V_obs[~np.isnan(V_obs)],
        s=10,
        color="red",
        alpha=0.3,
        label="Observations",
    )
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Particle filter on HH model (RMSE={rmse:.2f} mV)")
    plt.tight_layout()
    plt.savefig("results/particle_filter_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Créer un dossier pour les résultats s'il n'existe pas
    os.makedirs("results", exist_ok=True)

    # Charger les données
    print("📂 Chargement des données...")
    data = np.load("hh_data.npz", allow_pickle=True)

    V_true = data["V_true"]
    y_obs = data["y_obs"]
    dt = data["dt"].item()

    print(f"Données chargées:")
    print(f"   - V_true shape: {V_true.shape}")
    print(f"   - dt: {dt} ms")

    # Exécuter le filtre particulaire AVEC sauvegarde des particules
    print("Particle Filter Running")

    start_time = time.time()

    # On récupère aussi les particules
    V_filt, particles, weights = particle_filter(
        y_obs,
        N=300,
        dt=dt,
        I=10.0,
        sigma_dyn=0.3,
        sigma_obs=5.0,
        save_particles=True,
    )

    elapsed = time.time() - start_time
    print(f"✅ Filtrage terminé en {elapsed:.2f} secondes")

    # Sauvegarder les résultats
    print("\n💾 Sauvegarde des résultats...")
    np.savez(
        "results/particle_filter_results.npz",
        V_filt=V_filt,
        particles=particles,
        weights=weights,
        V_true=V_true,
        y_obs=y_obs,
        dt=dt,
        I=8.0,
        sigma_dyn=0.3,
        sigma_obs=5.0,
    )

    print("✅ Résultats sauvegardés dans:")
    print("   results/particle_filter_results.npz")
    print(f"   - particles shape: {particles.shape}")
    print(f"   - weights shape: {weights.shape}")

    # Calculer l'erreur
    mask = ~np.isnan(y_obs)
    rmse = np.sqrt(np.mean((V_filt[mask] - V_true[mask]) ** 2))
    print(f"\n📊 RMSE: {rmse:.2f} mV")

    # Visualisation
    t = np.arange(len(V_true)) * dt
    plt.figure(figsize=(12, 4))
    plt.plot(t, V_true, label="True voltage", lw=2)
    plt.plot(t, V_filt, label="Filtered estimate", lw=2, alpha=0.8)
    plt.scatter(
        t[~np.isnan(y_obs)],
        y_obs[~np.isnan(y_obs)],
        s=10,
        color="red",
        alpha=0.3,
        label="Observations",
    )
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Particle filter on HH model (RMSE={rmse:.2f} mV)")
    plt.tight_layout()
    plt.savefig("results/particle_filter_plot.png", dpi=150)
    plt.show()
    print("📈 Graphique sauvegardé: results/particle_filter_plot.png")
