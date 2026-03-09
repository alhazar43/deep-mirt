"""EAP (Expected A Posteriori) theta estimation given item parameters and responses.

This implements proper IRT theta estimation when item parameters are known,
following standard psychometric practice.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import simpson


def gpcm_probability(theta, alpha, beta):
    """Compute GPCM category probabilities.

    Args:
        theta: Student ability (scalar)
        alpha: Item discrimination (scalar)
        beta: Item thresholds (K-1,) array

    Returns:
        probs: Category probabilities (K,) array
    """
    K = len(beta) + 1

    # Compute cumulative logits
    logits = np.zeros(K)
    for k in range(1, K):
        logits[k] = logits[k-1] + alpha * (theta - beta[k-1])

    # Convert to probabilities
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    return probs


def log_likelihood_given_theta(theta, responses, questions, alpha_params, beta_params):
    """Compute log-likelihood of response pattern given theta.

    Args:
        theta: Student ability (scalar)
        responses: Response pattern (T,) array of category indices
        questions: Question IDs (T,) array
        alpha_params: Discrimination parameters (Q,) array
        beta_params: Threshold parameters (Q, K-1) array

    Returns:
        log_likelihood: log P(responses | theta, item_params)
    """
    log_likelihood = 0.0

    for r, q in zip(responses, questions):
        if q < 0 or q >= len(alpha_params):
            continue

        alpha = alpha_params[q]
        beta = beta_params[q]

        # Get probability of observed response
        probs = gpcm_probability(theta, alpha, beta)
        if 0 <= r < len(probs):
            log_likelihood += np.log(max(probs[int(r)], 1e-10))

    return log_likelihood


def estimate_theta_eap(responses, questions, alpha_params, beta_params,
                       prior_mean=0.0, prior_std=1.0, theta_grid=None):
    """Estimate theta using Expected A Posteriori (EAP) method.

    Args:
        responses: Response pattern (T,) array
        questions: Question IDs (T,) array
        alpha_params: Discrimination parameters (Q,) array
        beta_params: Threshold parameters (Q, K-1) array
        prior_mean: Prior mean for theta
        prior_std: Prior std for theta
        theta_grid: Grid of theta values for integration (default: -4 to 4)

    Returns:
        theta_eap: EAP estimate of theta
    """
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)  # Standard range

    # Compute log-posterior at each grid point
    log_posterior = np.zeros_like(theta_grid)

    for i, theta in enumerate(theta_grid):
        # Log prior: log N(prior_mean, prior_std^2)
        log_prior = norm.logpdf(theta, loc=prior_mean, scale=prior_std)

        # Log likelihood: log P(responses | theta)
        log_likelihood = log_likelihood_given_theta(
            theta, responses, questions, alpha_params, beta_params
        )

        # Log posterior = log prior + log likelihood
        log_posterior[i] = log_prior + log_likelihood

    # Convert to posterior (subtract max for numerical stability)
    log_posterior = log_posterior - log_posterior.max()
    posterior = np.exp(log_posterior)

    # Normalize posterior
    posterior = posterior / (simpson(posterior, x=theta_grid) + 1e-10)

    # EAP = E[theta | responses] = ∫ theta × p(theta | responses) dtheta
    theta_eap = simpson(theta_grid * posterior, x=theta_grid)

    return theta_eap
