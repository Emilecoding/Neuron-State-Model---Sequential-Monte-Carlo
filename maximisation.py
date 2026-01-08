import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import minimize
from generate_data import hh_step

def load_smoothed_results():
    """
    Charge les résultats du smoothing
    """
    print("📂 Chargement des résultats du smoothing...")
    
    try:
        data = np.load("results/smoothed_results.npz", allow_pickle=True)
        
        smoothed_particles = data["particles"]  # (T, N, 4)
        smoothed_weights = data["weights"]      # (T, N)
        V_smooth = data["V_smooth"]
        V_true = data["V_true"]
        y_obs = data["y_obs"]
        dt = data["dt"].item()
        I = data["I"].item()
        sigma_dyn = data["sigma_dyn"].item()
        
        print(f"   ✓ Données chargées:")
        print(f"     - particles: {smoothed_particles.shape}")
        print(f"     - weights: {smoothed_weights.shape}")
        print(f"     - T={len(V_smooth)}, dt={dt}, I={I}")
        
        return (smoothed_particles, smoothed_weights, V_smooth, 
                V_true, y_obs, dt, I, sigma_dyn)
        
    except FileNotFoundError:
        print("   ❌ Fichier non trouvé: results/smoothed_results.npz")
        print("   Exécutez d'abord: python smoothing.py")
        exit(1)
    except KeyError as e:
        print(f"   ❌ Clé manquante: {e}")
        print("   Vérifiez le fichier .npz")
        exit(1)


def compute_sigma_obs(smoothed_particles, y_obs, smoothed_weights):
    """
    Mise à jour de sigma_obs (bruit d'observation)
    
    Parameters
    ----------
    smoothed_particles : array (T, N, 4)
        Particules lissées
    y_obs : array (T,)
        Observations
    smoothed_weights : array (T, N)
        Poids lissés
    """
    obs_mask = ~np.isnan(y_obs)
    
    if np.sum(obs_mask) == 0:
        return 5.0  # Valeur par défaut
    
    # Moyenne pondérée des erreurs
    weighted_errors = 0
    total_weight = 0
    
    for t in np.where(obs_mask)[0]:
        for i in range(smoothed_weights.shape[1]):
            error = (y_obs[t] - smoothed_particles[t, i, 0])**2
            weight = smoothed_weights[t, i]
            weighted_errors += weight * error
            total_weight += weight
    
    sigma_obs_new = np.sqrt(weighted_errors / total_weight)
    
    return sigma_obs_new


def compute_sigma_dyn(smoothed_particles, smoothed_weights, dt, I):
    """
    Mise à jour de sigma_dyn (bruit de dynamique)
    """
    T, N, _ = smoothed_particles.shape
    weighted_errors = 0
    total_weight = 0
    
    for t in range(1, T):
        for i in range(N):
            # Prédiction à partir de l'état précédent
            pred = hh_step(smoothed_particles[t-1, i], I, dt, noise_std=0.0)
            
            # Erreur de prédiction
            error = np.sum((smoothed_particles[t, i] - pred)**2)
            
            # Poids = produit des poids aux temps t et t-1
            weight = smoothed_weights[t, i] * smoothed_weights[t-1, i]
            
            weighted_errors += weight * error
            total_weight += weight
    
    # Normaliser par le nombre de dimensions d'état
    sigma_dyn_new = np.sqrt(weighted_errors / total_weight / smoothed_particles.shape[2])
    
    return sigma_dyn_new


def estimate_hh_parameters(smoothed_particles, smoothed_weights, dt, I, sigma_dyn):
    """
    Estimation des paramètres HH via maximum de vraisemblance
    """
    T, N, _ = smoothed_particles.shape
    
    def hh_currents(V, m, h, n, params):
        """Calcule les courants HH avec paramètres donnés"""
        gNa, gK, gL, ENa, EK, EL = params
        
        INa = gNa * m**3 * h * (V - ENa)
        IK = gK * n**4 * (V - EK)
        IL = gL * (V - EL)
        
        return INa, IK, IL
    
    def negative_log_likelihood(params):
        """Fonction de vraisemblance négative à minimiser"""
        gNa, gK, gL, ENa, EK, EL = params
        
        log_lik = 0
        
        for t in range(1, T):
            for i in range(N):
                # État courant
                V, m, h, n = smoothed_particles[t, i]
                V_prev, m_prev, h_prev, n_prev = smoothed_particles[t-1, i]
                
                # Courants avec paramètres courants
                INa, IK, IL = hh_currents(V, m, h, n, params)
                
                # Prédiction de voltage
                dV_pred = (I - INa - IK - IL) / 1.0  # C=1
                V_pred = V_prev + dt * dV_pred
                
                # Contribution à la log-vraisemblance
                error = (V - V_pred)**2
                weight = smoothed_weights[t, i]
                log_lik -= weight * error / (2 * sigma_dyn**2)
        
        return -log_lik
    
    # Initial guess (valeurs standard de HH)
    params_init = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4]
    
    # Bornes pour les paramètres (physiquement plausibles)
    bounds = [
        (50, 200),    # gNa
        (10, 60),     # gK
        (0.1, 1.0),   # gL
        (40, 60),     # ENa
        (-90, -60),   # EK
        (-70, -40)    # EL
    ]
    
    print("   🔍 Optimisation des paramètres HH...")
    start_opt = time.time()
    
    result = minimize(
        negative_log_likelihood,
        params_init,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 50, 'disp': False}
    )
    
    opt_time = time.time() - start_opt
    
    if result.success:
        print(f"   ✓ Optimisation réussie ({opt_time:.1f}s)")
        params_opt = result.x
    else:
        print(f"   ⚠  Optimisation non convergée, utilisation des valeurs initiales")
        params_opt = params_init
    
    return params_opt


def plot_parameter_convergence(params_history, true_values=None):
    """
    Trace l'évolution des paramètres au cours des itérations EM
    """
    os.makedirs("figures", exist_ok=True)
    
    param_names = ['gNa', 'gK', 'gL', 'ENa', 'EK', 'EL']
    n_params = len(param_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, ax) in enumerate(zip(param_names, axes)):
        values = [p[idx] for p in params_history]
        iterations = range(len(values))
        
        ax.plot(iterations, values, 'o-', linewidth=2, markersize=8)
        
        if true_values is not None and idx < len(true_values):
            ax.axhline(y=true_values[idx], color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Valeur vraie')
        
        ax.set_xlabel('Itération EM')
        ax.set_ylabel(name)
        ax.set_title(f'Convergence de {name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("figures/parameter_convergence.png", dpi=150, bbox_inches='tight')
    plt.savefig("figures/parameter_convergence.pdf")
    plt.show()
    
    return fig


def plot_currents_comparison(smoothed_particles, params_estimated, params_true, dt):
    """
    Compare les courants calculés avec les paramètres estimés vs vrais
    """
    T, N, _ = smoothed_particles.shape
    
    # Prendre un échantillon de particules
    sample_idx = np.random.choice(N, size=min(20, N), replace=False)
    
    # Calculer les courants moyens
    INa_est_list, IK_est_list = [], []
    INa_true_list, IK_true_list = [], []
    
    for t in range(0, T, 10):  # Sous-échantillonnage temporel
        for i in sample_idx:
            V, m, h, n = smoothed_particles[t, i]
            
            # Courants avec paramètres estimés
            gNa_est, gK_est, gL_est, ENa_est, EK_est, EL_est = params_estimated
            INa_est = gNa_est * m**3 * h * (V - ENa_est)
            IK_est = gK_est * n**4 * (V - EK_est)
            
            # Courants avec paramètres vrais
            gNa_true, gK_true, gL_true = 120.0, 36.0, 0.3
            ENa_true, EK_true, EL_true = 50.0, -77.0, -54.4
            INa_true = gNa_true * m**3 * h * (V - ENa_true)
            IK_true = gK_true * n**4 * (V - EK_true)
            
            INa_est_list.append(INa_est)
            IK_est_list.append(IK_est)
            INa_true_list.append(INa_true)
            IK_true_list.append(IK_true)
    
    # Créer le plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Courant Na
    axes[0].scatter(INa_true_list, INa_est_list, alpha=0.5, s=10)
    axes[0].plot([min(INa_true_list), max(INa_true_list)],
                [min(INa_true_list), max(INa_true_list)], 
                'r--', alpha=0.7, label='y=x')
    axes[0].set_xlabel('Courant Na réel (nA)')
    axes[0].set_ylabel('Courant Na estimé (nA)')
    axes[0].set_title('Courant sodium')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Courant K
    axes[1].scatter(IK_true_list, IK_est_list, alpha=0.5, s=10, color='orange')
    axes[1].plot([min(IK_true_list), max(IK_true_list)],
                [min(IK_true_list), max(IK_true_list)], 
                'r--', alpha=0.7, label='y=x')
    axes[1].set_xlabel('Courant K réel (nA)')
    axes[1].set_ylabel('Courant K estimé (nA)')
    axes[1].set_title('Courant potassium')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/currents_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    print("="*60)
    print("M-STEP COMPLET - ESTIMATION DES PARAMÈTRES")
    print("="*60)
    
    # 1. Charger les données
    (smoothed_particles, smoothed_weights, V_smooth, 
     V_true, y_obs, dt, I, sigma_dyn) = load_smoothed_results()
    
    # Réduire la taille pour tests rapides
    MAX_POINTS = 500
    if len(V_smooth) > MAX_POINTS:
        print(f"\n⚠  Utilisation des {MAX_POINTS} premiers points pour test rapide")
        smoothed_particles = smoothed_particles[:MAX_POINTS]
        smoothed_weights = smoothed_weights[:MAX_POINTS]
        V_smooth = V_smooth[:MAX_POINTS]
        V_true = V_true[:MAX_POINTS]
        y_obs = y_obs[:MAX_POINTS]
    
    print(f"\n📊 Statistiques des données:")
    print(f"   - Nombre de points: {len(V_smooth)}")
    print(f"   - Observations: {np.sum(~np.isnan(y_obs))}")
    print(f"   - Voltage range: [{V_smooth.min():.1f}, {V_smooth.max():.1f}] mV")
    
    # 2. Estimation des paramètres de bruit
    print("\n[1/3] Estimation des paramètres de bruit...")
    
    sigma_obs_new = compute_sigma_obs(smoothed_particles, y_obs, smoothed_weights)
    sigma_dyn_new = compute_sigma_dyn(smoothed_particles, smoothed_weights, dt, I)
    
    print(f"   ✓ sigma_obs: {sigma_obs_new:.3f} mV")
    print(f"   ✓ sigma_dyn: {sigma_dyn_new:.3f}")
    
    # 3. Estimation des paramètres HH
    print("\n[2/3] Estimation des paramètres HH...")
    
    params_estimated = estimate_hh_parameters(
        smoothed_particles, smoothed_weights, dt, I, sigma_dyn_new
    )
    
    param_names = ['gNa', 'gK', 'gL', 'ENa', 'EK', 'EL']
    true_values = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4]
    
    print("\n   Paramètres estimés vs vrais:")
    for name, est, true in zip(param_names, params_estimated, true_values):
        error_pct = 100 * abs(est - true) / true
        print(f"     {name}: {est:6.2f} (vrai: {true:6.2f}) | erreur: {error_pct:.1f}%")
    
    # 4. Sauvegarder les résultats
    print("\n[3/3] Sauvegarde des résultats...")
    os.makedirs("results", exist_ok=True)
    
    results_dict = {
        'sigma_obs': sigma_obs_new,
        'sigma_dyn': sigma_dyn_new,
        'params_estimated': params_estimated,
        'param_names': param_names,
        'true_values': true_values,
        'V_smooth': V_smooth,
        'V_true': V_true,
        'y_obs': y_obs,
        'dt': dt,
        'I': I
    }
    
    np.savez("results/mstep_results.npz", **results_dict)
    
    print(" Résultats sauvegardés dans: results/mstep_results.npz")

if __name__ == "__main__":

    main()