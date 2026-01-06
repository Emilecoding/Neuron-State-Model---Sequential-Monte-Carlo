from data import *
import particles
from particles import state_space_models as ssm
from particles.collectors import Moments
import particles.distributions as dists



def particle_filter(y : Vector, transition_matrix : Matrix, mu : Vector, sigma : float, N = 500) -> tuple[Vector, float]:
    T = len(y)
    n_state = len(mu)

    # Initialisation
    particles = np.random.choice(n_state, size = N)
    weights = np.ones(N) / N
    x_estimated = np.zeros(T)
    log_likelihood = 0.0

    # Estimation
    for t in range(T):

        particles = np.array([np.random.choice(n_state, p = transition_matrix[p]) for p in particles])

        weights *= np.exp(-0.5 * (y[t] - mu[particles])**2 / sigma**2)
        weights /= np.sum(weights)
    
        x_estimated[t] = np.sum(weights * particles)

        ll = np.mean(np.exp(-0.5*((y[t] - mu[particles])**2) / sigma**2))
        log_likelihood += np.log(ll + 1e-12)

        ess = 1 / np.sum(weights**2)
        if ess < N/2:
            idx = np.random.choice(N, size = N, p = weights)
            particles = particles[idx]
            weights = np.ones(N) / N 

    return x_estimated, log_likelihood


class TwoStateGaussianHMM(ssm.StateSpaceModel):
    def PX0(self):
        return dists.Categorical(p=self.pi_0)

    def PX(self, t, xp):
        return dists.Categorical(p=self.transition_matrix[xp])

    def PY(self, t, xp, x):
        return dists.Normal(loc=self.mu[x], scale=self.sigma)


def run_particle_filter(y_obs, transition_matrix, pi_0, mu, sigma, N=2000, resampling="stratified", seed=123):
    np.random.seed(seed)

    model = TwoStateGaussianHMM(transition_matrix=transition_matrix, pi_0=pi_0, mu=mu, sigma=sigma)

    # Bootstrap FK model + SMC (comme dans la doc)
    fk_model = ssm.Bootstrap(ssm=model, data=y_obs)
    pf = particles.SMC(
        fk=fk_model,
        N=N,
        resampling=resampling,
        collect=[Moments()],      
        store_history=True
    )
    pf.run()

    p_open = np.array([m["mean"] for m in pf.summaries.moments])

    return pf, p_open


def demo():
    # données synthétiques
    x = simulate_markov_chain(transition_matrix, pi_0, T)
    y = generate_observations(x, mu, sigma)
    x_estimated, ll = particle_filter(y, transition_matrix, mu, sigma, N=1000)


    # PF
    pf, p_open = run_particle_filter(y, transition_matrix, pi_0, mu, sigma, N=1000)

    # affichage (sur une fenêtre)
    K = 400
    t = np.arange(K)

    plt.figure(figsize=(12, 7))

    plt.subplot(4, 1, 1)
    plt.plot(t, x[:K], lw=1)
    plt.yticks([0, 1])
    plt.title("État caché (vrai)")
    plt.ylabel("x_t")

    plt.subplot(4, 1, 2)
    plt.plot(t, y[:K], lw=0.8)
    for m in mu:
        plt.axhline(m, ls="--", alpha=0.5)
    plt.title("Observations")
    plt.ylabel("y_t")

    plt.subplot(4, 1, 3)
    plt.plot(t, p_open[:K], lw=1)
    plt.ylim(-0.05, 1.05)
    plt.title("Filtrage particulaire: P(x_t=1 | y_0:t) (approx.)")
    plt.ylabel("proba état 1")
    plt.xlabel("temps")

    plt.subplot(4,1,4)
    plt.plot(x_estimated[:K], label="Particle filter", lw=1)
    plt.legend()
    plt.title("Particle filter - 500 first points")

    plt.tight_layout()
    plt.show()

    print("log-likelihood estimate (logLt) =", pf.logLt)


if __name__ == "__main__":
    demo()