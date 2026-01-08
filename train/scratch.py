import jax
import jax.numpy as jnp
from jax.scipy.special import logit, expit # expit is sigmoid

def verify_equivalence():
    key = jax.random.PRNGKey(42)
    N_SAMPLES = 1_000_000
    
    # 1. Choose an arbitrary logit x
    # Let's pick x = 0.5. 
    # The expected probability is sigmoid(0.5) approx 0.6224
    x_logit = 0.8
    expected_prob = expit(x_logit)
    
    print(f"Input Logit: {x_logit}")
    print(f"Expected Probability (Sigmoid): {expected_prob:.4f}")
    print("-" * 30)

    # 2. Generate Logistic Noise
    # log(u) - log(1-u)
    u = jax.random.uniform(key, shape=(N_SAMPLES,))
    eps = jnp.log(u / (1.0 - u))

    # 3. Perform the "Round(x + noise)" operation
    # This is equivalent to checking if (x + noise) > 0
    # or round(sigmoid(x + noise))
    samples = (x_logit + eps) > 0

    # 4. Calculate Empirical Mean
    empirical_prob = jnp.mean(samples)
    
    print(f"Empirical Mean (over {N_SAMPLES} samples): {empirical_prob:.4f}")
    print(f"Difference: {jnp.abs(empirical_prob - expected_prob):.6f}")

if __name__ == "__main__":
    verify_equivalence()