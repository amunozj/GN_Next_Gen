# ADR-001: Python + JAX/NumPyro, conda environments (GPU default, CPU variant)

- Status: Accepted (2026-10-28, PI)
- Context: The joint latent-count model needs exact forward filtering over ~150k daily steps, vectorized observer
  updates, and batched chains; the PI chose a differentiable probabilistic-programming stack.
- Decision: Python with JAX and NumPyro (plus optax, arviz; dynamax/blackjax/BayesFlow evaluated per step). Environments
  are conda/mamba: `environment.yml` (GPU, CUDA 12 JAX wheels) is the default, `environment-cpu.yml` the CPU variant;
  `environment-legacy.yml` exists only to run legacy baselines.
- Consequences: JAX GPU wheels are Linux-only, so on Windows use **WSL2** (native Windows is CPU-only). Accumulate
  log-likelihoods in float64 (`jax.config.update("jax_enable_x64", True)` where needed). Hand-written forward filters are
  preferred over NumPyro automatic enumeration inside long `scan`s (known fragility; see research report).
