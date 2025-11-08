"""
Bayesian 2PL / Ideal-point estimator on a bounded ability grid θ∈[0,1].

Update (batch API):
- Call `estimate(Y, b, ..., likelihood='2pl'|'ideal', a=..., kappa=..., gamma=...)`.
- That call *updates and persists* the posterior, then returns summaries.

Models
------
- 2PL    : p_i(θ) = σ(a_i (θ - b_i)),         a_i > 0
- Ideal  : p_i(θ) = σ(γ_i - κ_i (θ - b_i)^2), κ_i > 0

Fast, vectorized, no samplers. A batch update costs O(k*G) with k=batch size, G=grid points.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List
import numpy as np
from math import lgamma
import json
import plotly.graph_objects as go
import json, uuid
import plotly.io as pio

class Bayesian2PL:
    """Bayesian estimator for user ability under 2PL or Ideal-point likelihoods on θ∈[0,1]."""

    def __init__(
        self,
        *,
        grid_points: int = 1001,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-9,
        default_a: float = 4.0,
    ) -> None:
        if grid_points < 101:
            raise ValueError("grid_points should be at least 101 for reasonable accuracy")
        self.grid_points = int(grid_points)
        self.grid = np.linspace(0.0, 1.0, self.grid_points)
        self.eps = float(eps)
        self.default_a = float(default_a)

        # Prior log-pdf on the grid: Beta(alpha, beta)
        self._set_prior(alpha, beta)

        # Running state
        self.history: List[Dict[str, Any]] = []  # batches with params + likelihood tag
        self._log_posterior = self.log_prior.copy()
        self._posterior: Optional[np.ndarray] = None
        self._cdf: Optional[np.ndarray] = None
        self._normalized = False

        self._n_obs: int = 0
        self._sum_reward: float = 0.0

    # ----------------------------- Internal helpers -----------------------------
    def _set_prior(self, alpha: float, beta: float) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")
        x = np.clip(self.grid, self.eps, 1.0 - self.eps)
        const = (lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta))
        self.log_prior = ((alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x) + const)

    def _normalize(self) -> None:
        m = np.max(self._log_posterior)
        w = np.exp(self._log_posterior - m)
        Z = np.trapezoid(w, self.grid)
        self._posterior = w / (Z + 1e-300)
        self._cdf = np.cumsum(self._posterior)
        self._cdf /= self._cdf[-1]
        self._normalized = True

    def _validate_batch_common(self, Y: Sequence[int | bool], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(Y, dtype=int).ravel()
        B = np.asarray(b, dtype=float).ravel()
        if y.size == 0 or B.size == 0 or y.size != B.size:
            raise ValueError("Y and b must be non-empty 1D arrays of the same length")
        if np.any((B < 0.0) | (B > 1.0)):
            raise ValueError("All b_i must lie in [0, 1]")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Y must contain only 0/1")
        return y, B

    def _batch_loglik_on_grid(
        self,
        *,
        y: np.ndarray,
        b: np.ndarray,
        likelihood: str,
        a: Optional[np.ndarray] = None,
        kappa: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-grid log-likelihood for a batch (shape: (G,)), for chosen likelihood."""
        G = self.grid.size
        if likelihood == "2pl":
            A = self.default_a * np.ones_like(b) if a is None else np.asarray(a, dtype=float).ravel()
            if A.size != b.size: raise ValueError("a must match b length")
            if np.any(A <= 0.0): raise ValueError("all a_i must be > 0")
            logits = A[:, None] * (self.grid[None, :] - b[:, None])  # (k,G)
        elif likelihood == "ideal":
            K = np.full_like(b, 50.0, dtype=float) if kappa is None else np.asarray(kappa, dtype=float).ravel()
            Gm = np.zeros_like(b, dtype=float)     if gamma is None else np.asarray(gamma, dtype=float).ravel()
            if K.size != b.size or Gm.size != b.size:
                raise ValueError("kappa and gamma must match b length")
            if np.any(K <= 0.0): raise ValueError("all kappa_i must be > 0")
            logits = Gm[:, None] - K[:, None] * (self.grid[None, :] - b[:, None])**2
        else:
            raise ValueError("likelihood must be '2pl' or 'ideal'")

        logits = np.clip(logits, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, self.eps, 1.0 - self.eps)
        ll = (y[:, None] * np.log(p) + (1 - y)[:, None] * np.log(1.0 - p)).sum(axis=0)  # (G,)
        return ll

    # ----------------------------- Public API -----------------------------------
    def reset(self, *, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Reset to the prior; clears all accumulated batches and posterior."""
        self._set_prior(alpha, beta)
        self._log_posterior = self.log_prior.copy()
        self.history.clear()
        self._posterior = None
        self._cdf = None
        self._normalized = False
        self._n_obs = 0
        self._sum_reward = 0.0

    def update(
        self,
        Y: Sequence[int | bool],
        b: Sequence[float],
        a: Optional[Sequence[float]] = None,
        *,
        likelihood: str = "2pl",
        kappa: Optional[Sequence[float]] = None,
        gamma: Optional[Sequence[float]] = None,
    ) -> None:
        """Update posterior with a *batch* under the chosen likelihood ('2pl' or 'ideal')."""
        y, B = self._validate_batch_common(Y, b)
        self._n_obs += int(y.size)
        self._sum_reward += float(y.sum())        
        ll = self._batch_loglik_on_grid(
            y=y, b=B, likelihood=likelihood.lower(),
            a=None if a is None else np.asarray(a, dtype=float).ravel(),
            kappa=None if kappa is None else np.asarray(kappa, dtype=float).ravel(),
            gamma=None if gamma is None else np.asarray(gamma, dtype=float).ravel(),
        )
        self._log_posterior = self._log_posterior + ll
        self._normalized = False

        # audit trail
        rec: Dict[str, Any] = {"y": y.tolist(), "b": B.tolist(), "likelihood": likelihood.lower()}
        if likelihood.lower() == "2pl":
            rec["a"] = (self.default_a * np.ones_like(B) if a is None else np.asarray(a, dtype=float)).tolist()
        else:
            rec["kappa"] = (np.full_like(B, 50.0, dtype=float) if kappa is None else np.asarray(kappa, dtype=float)).tolist()
            rec["gamma"] = (np.zeros_like(B, dtype=float) if gamma is None else np.asarray(gamma, dtype=float)).tolist()
        self.history.append(rec)

    def posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (grid, posterior density) on θ ∈ [0,1]."""
        if not self._normalized:
            self._normalize()
        return self.grid.copy(), self._posterior.copy()  # type: ignore

    def estimate(
        self,
        Y: Optional[Sequence[int | bool]] = None,
        b: Optional[Sequence[float]] = None,
        a: Optional[Sequence[float]] = None,
        *,
        likelihood: str = "2pl",
        kappa: Optional[Sequence[float]] = None,
        gamma: Optional[Sequence[float]] = None,
        method: str = "mean",
        ci_level: float = 0.90,
        return_posterior: bool = False,
    ) -> Dict[str, Any]:
        """(Optionally) update with a batch, then return current θ summaries."""
        if (Y is None) ^ (b is None):
            raise ValueError("Provide both Y and b, or neither.")
        if Y is not None and b is not None:
            self.update(Y, b, a, likelihood=likelihood, kappa=kappa, gamma=gamma)

        grid, post = self.posterior()
        if method == "mean":
            theta = float(np.trapezoid(grid * post, grid))
        elif method == "map":
            theta = float(grid[np.argmax(post)])
        elif method == "median":
            cdf = np.cumsum(post); cdf /= cdf[-1]
            theta = float(np.interp(0.5, cdf, grid))
        else:
            raise ValueError("method must be one of {'mean','map','median'}")

        # equal-tailed CI
        cdf = np.cumsum(post); cdf /= cdf[-1]
        lo = float(np.interp((1.0 - ci_level)/2.0, cdf, grid))
        hi = float(np.interp(1.0 - (1.0 - ci_level)/2.0, cdf, grid))
        out = {"theta": theta, "method": method, "ci": (lo, hi)}
        if return_posterior:
            out.update({"grid": grid, "posterior": post})
        return out

    def predictive_prob_for(
        self,
        b: float,
        a: Optional[float] = None,
        *,
        likelihood: str = "2pl",
        kappa: Optional[float] = None,
        gamma: float = 0.0,
        use_posterior: bool = True,
        theta: Optional[float] = None,
    ) -> float:
        """Posterior-predictive success for a single (b, params) under chosen likelihood."""
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0,1]")
        lik = likelihood.lower()
        if lik == "2pl":
            a_val = self.default_a if a is None else float(a)
            if a_val <= 0: raise ValueError("a must be > 0")
            if use_posterior:
                grid, post = self.posterior()
                logits = np.clip(a_val * (grid - b), -50.0, 50.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                return float(np.trapezoid(p * post, grid))
            else:
                if theta is None:
                    theta = self.estimate(method="mean")["theta"]
                return float(1.0 / (1.0 + np.exp(-a_val * (theta - b))))
        elif lik == "ideal":
            k_val = 50.0 if kappa is None else float(kappa)
            if k_val <= 0: raise ValueError("kappa must be > 0")
            if use_posterior:
                grid, post = self.posterior()
                logits = np.clip(gamma - k_val * (grid - b)**2, -50.0, 50.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                return float(np.trapezoid(p * post, grid))
            else:
                if theta is None:
                    theta = self.estimate(method="mean")["theta"]
                val = gamma - k_val * (float(theta) - b)**2
                return float(1.0 / (1.0 + np.exp(-np.clip(val, -50.0, 50.0))))
        else:
            raise ValueError("likelihood must be '2pl' or 'ideal'")

    def running_reward_stats(self) -> Dict[str, float]:
        """Return cumulative stats of observed binary rewards so far."""
        avg = (self._sum_reward / self._n_obs) if self._n_obs > 0 else float("nan")
        return {
            "trials": float(self._n_obs),
            "reward_sum": float(self._sum_reward),
            "avg_reward": float(avg),
        }

    # =====================[ Bandit storage/helpers ]=====================

    # =====================[ Bandit storage/helpers ]=====================
    def _ensure_bandit_storage(self):
        if not hasattr(self, "_bandit_stats"):
            # key -> {"S": float, "N": float, "alpha": float, "beta": float, "t": int}
            self._bandit_stats: Dict[tuple, Dict[str, float]] = {}
            self._bandit_t: int = 0  # logical time (ticks when you call next_item)

    def _arm_key(self, *, likelihood: str, b: float,
                a: Optional[float], kappa: Optional[float], gamma: Optional[float]) -> tuple:
        """Stable key for an arm; rounded to avoid float noise."""
        if likelihood == "2pl":
            return ("2pl", round(float(b), 6),
                    round(float(a if a is not None else self.default_a), 6))
        else:
            return ("ideal", round(float(b), 6),
                    round(float(kappa if kappa is not None else 50.0), 6),
                    round(float(gamma if gamma is not None else 0.0), 6))

    def bandit_record_outcome(
        self,
        *,
        b: float,
        likelihood: str = "2pl",
        a: Optional[float] = None,
        kappa: Optional[float] = None,
        gamma: float = 0.0,
        y: int,
        discount: float = 0.97,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        """
        Log one binary reward y∈{0,1} for the chosen arm with exponential discounting.
        Updates BOTH:
        - discounted counts (S, N)  -> used by 'discounted_ucb'
        - discounted Beta posterior (alpha, beta) -> used by 'beta_ts'/'beta_ucb'
        """
        self._ensure_bandit_storage()
        lam = float(discount)
        if not (0 < lam <= 1):
            raise ValueError("discount must be in (0,1]")

        # NOTE: we do NOT advance time here; next_item() advances _bandit_t.
        t_now = self._bandit_t

        key = self._arm_key(likelihood=likelihood.lower(), b=b, a=a, kappa=kappa, gamma=gamma)
        st = self._bandit_stats.get(key)
        if st is None:
            st = {"S": 0.0, "N": 0.0, "alpha": float(alpha0), "beta": float(beta0), "t": t_now}

        # decay since last update
        dt = t_now - st["t"]
        decay = lam ** max(dt, 0)

        st["S"]     = st["S"]     * decay + float(y)
        st["N"]     = st["N"]     * decay + 1.0
        st["alpha"] = st["alpha"] * decay + float(y)
        st["beta"]  = st["beta"]  * decay + float(1 - y)
        st["t"]     = t_now

        self._bandit_stats[key] = st

    #--Sampling Strategy----

    def next_item(
        self,
        candidates_b: Sequence[float],
        candidates_a: Optional[Sequence[float]] = None,
        *,
        likelihood: str = "2pl",
        ideal_kappa: Optional[Sequence[float]] = None,
        ideal_gamma: Optional[Sequence[float]] = None,
        # which engine to *use* at runtime
        engine: str = "bayes",                  # {"bayes","bandit","hybrid"}
        # Bayes engine policy (+params)
        bayes_policy: str = "thompson",         # {"thompson","greedy","ucb","bayesucb","max_info","closest_b"}
        ucb_kappa: float = 1.0,
        quantile: Optional[float] = None,
        delta: Optional[float] = None,
        step_t: Optional[int] = None,
        alpha: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        # Bandit params
        bandit_strategy: str = "discounted_ucb",  # {"discounted_ucb","beta_ts","beta_ucb","beta_mean"}
        bandit_discount: float = 0.97,
        bandit_c: float = 2.0,
        bandit_alpha0: float = 1.0,
        bandit_beta0: float = 1.0,
        # Hybrid mixing
        hybrid_eta: float = 0.5,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute BOTH:
        (A) Bayesian CAT selection (using `bayes_policy`)
        (B) Bandit selection (choose via `bandit_strategy`),
        then return the choice according to `engine`.
        """
        import numpy as np

        # ---------------- sanity + per-likelihood params ----------------
        lik = likelihood.lower()
        B = np.asarray(candidates_b, dtype=float).ravel()
        if B.size == 0:
            raise ValueError("candidates_b must be non-empty")
        if np.any((B < 0.0) | (B > 1.0)):
            raise ValueError("All candidate b must be in [0,1]")

        if lik == "2pl":
            if candidates_a is None:
                A = np.full_like(B, self.default_a, dtype=float)
            else:
                A = np.asarray(candidates_a, dtype=float).ravel()
                if A.size != B.size: raise ValueError("candidates_a must match candidates_b")
                if np.any(A <= 0.0): raise ValueError("all a>0")
            K = Gm = None
        elif lik == "ideal":
            K = np.full_like(B, 50.0, dtype=float) if ideal_kappa is None else np.asarray(ideal_kappa, dtype=float).ravel()
            Gm = np.zeros_like(B, dtype=float)     if ideal_gamma is None else np.asarray(ideal_gamma, dtype=float).ravel()
            if K.size != B.size or Gm.size != B.size:
                raise ValueError("ideal_kappa and ideal_gamma must match candidates_b length")
            if np.any(K <= 0.0): raise ValueError("all kappa>0")
            A = None
        else:
            raise ValueError("likelihood must be '2pl' or 'ideal'")

        # ======================= (A) BAYESIAN ENGINE =======================
        grid, post = self.posterior()
        if lik == "2pl":
            logits = np.clip(A[:, None] * (grid[None, :] - B[:, None]), -50.0, 50.0)
        else:
            logits = np.clip(Gm[:, None] - K[:, None] * (grid[None, :] - B[:, None])**2, -50.0, 50.0)
        P = 1.0 / (1.0 + np.exp(-logits))
        P = np.clip(P, self.eps, 1.0 - self.eps)

        expected_success = np.trapezoid(P * post[None, :], grid, axis=1)
        expected_sq      = np.trapezoid((P**2) * post[None, :], grid, axis=1)
        var_success      = np.maximum(expected_sq - expected_success**2, 0.0)
        std_success      = np.sqrt(var_success)

        policy = bayes_policy.lower()
        theta_sample = None
        theta_q = None
        q_used = None

        if policy == "greedy":
            bayes_scores = expected_success
        elif policy == "ucb":
            bayes_scores = expected_success + float(ucb_kappa) * std_success
        elif policy == "thompson":
            Gsz = grid.size
            delta_h = 1.0 / (Gsz - 1)
            trap_w = np.full(Gsz, delta_h); trap_w[0] *= 0.5; trap_w[-1] *= 0.5
            w = post * trap_w; w = w / (w.sum() + 1e-300)
            if rng is None: rng = np.random.default_rng()
            g = int(rng.choice(Gsz, p=w))
            theta_sample = float(grid[g])
            if lik == "2pl":
                bayes_scores = 1.0 / (1.0 + np.exp(-np.clip(A * (theta_sample - B), -50.0, 50.0)))
            else:
                bayes_scores = 1.0 / (1.0 + np.exp(-np.clip(Gm - K * (theta_sample - B)**2, -50.0, 50.0)))
        elif policy == "bayesucb":
            cdf_theta = np.cumsum(post); cdf_theta /= cdf_theta[-1]
            if quantile is not None:
                q_used = float(quantile)
            elif delta is not None:
                q_used = 1.0 - float(delta)
            else:
                answered = sum(len(h.get("y", [])) for h in self.history)
                t_sched = int(step_t) if step_t is not None else (answered + 1)
                t_sched = max(t_sched, 2)
                q_used = 1.0 - 1.0 / (t_sched ** float(alpha))
            q_used = float(np.clip(q_used, 1e-6, 1.0 - 1e-6))

            Gsz = grid.size
            delta_h = 1.0 / (Gsz - 1)
            trap_w = np.full(Gsz, delta_h); trap_w[0] *= 0.5; trap_w[-1] *= 0.5
            w = post * trap_w; w = w / (w.sum() + 1e-300)

            if lik == "2pl":
                theta_q = float(np.interp(q_used, cdf_theta, grid))
                bayes_scores = 1.0 / (1.0 + np.exp(-np.clip(A * (theta_q - B), -50.0, 50.0)))
            else:
                J = B.size
                qvals = np.empty(J, dtype=float)
                for j in range(J):
                    pj = P[j]
                    order = np.argsort(pj)
                    cw = np.cumsum(w[order])
                    qvals[j] = float(np.interp(q_used, cw, pj[order]))
                bayes_scores = qvals
        elif policy == "max_info":
            if lik == "2pl":
                dlogit = A[:, None] * np.ones_like(P)
            else:
                dlogit = -2.0 * K[:, None] * (grid[None, :] - B[:, None])
            info = (dlogit ** 2) * P * (1.0 - P)
            bayes_scores = np.trapezoid(info * post[None, :], grid, axis=1)
        elif policy == "closest_b":
            theta_mean = float(np.trapezoid(grid * post, grid))
            bayes_scores = -np.abs(B - theta_mean)
        else:
            raise ValueError("bayes_policy must be one of {'thompson','greedy','ucb','bayesucb','max_info','closest_b'}")

        idx_bayes = int(np.argmax(bayes_scores))

        # ======================= (B) BANDIT ENGINE =======================
        self._ensure_bandit_storage()
        # advance bandit time on *selection* calls (record_outcome uses current time)
        self._bandit_t += 1
        t_now = self._bandit_t
        lam = float(bandit_discount)
        if not (0 < lam <= 1):
            raise ValueError("bandit_discount must be in (0,1]")

        # helper: normal inverse CDF (Acklam approximation; max error ~4e-4)
        def _norm_ppf(p: float) -> float:
            # coefficients
            a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
            b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01]
            c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
            d = [7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
                3.754408661907416e+00]
            p = float(p)
            if not (0.0 < p < 1.0):
                return np.nan
            # break-points
            plow  = 0.02425
            phigh = 1 - plow
            if p < plow:
                q = np.sqrt(-2*np.log(p))
                return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
            if phigh < p:
                q = np.sqrt(-2*np.log(1-p))
                return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
            q = p - 0.5
            r = q*q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

        J = B.size
        Neff = np.empty(J); Seff = np.empty(J)
        alpha_eff = np.empty(J); beta_eff = np.empty(J)

        for j in range(J):
            key = self._arm_key(likelihood=lik, b=B[j],
                                a=(A[j] if A is not None else None),
                                kappa=(K[j] if K is not None else None),
                                gamma=(Gm[j] if Gm is not None else None))
            st = self._bandit_stats.get(key)
            if st is None:
                # unseen arm: start at prior
                Neff[j] = 0.0; Seff[j] = 0.0
                alpha_eff[j] = float(bandit_alpha0)
                beta_eff[j]  = float(bandit_beta0)
            else:
                dt = t_now - st["t"]
                decay = lam ** max(dt, 0)
                Neff[j] = st["N"] * decay
                Seff[j] = st["S"] * decay
                alpha_eff[j] = st["alpha"] * decay if st["alpha"] > 0 else float(bandit_alpha0)
                beta_eff[j]  = st["beta"]  * decay if st["beta"]  > 0 else float(bandit_beta0)

        # choose bandit strategy
        bstr = bandit_strategy.lower()
        if bstr == "discounted_ucb":
            mean_hat = (bandit_alpha0 + Seff) / (bandit_alpha0 + bandit_beta0 + Neff + 1e-300)
            bonus = np.sqrt(bandit_c * np.log(1.0 + t_now) / np.maximum(Neff, 1e-9))
            bandit_scores = mean_hat + bonus
        elif bstr == "beta_mean":
            bandit_scores = alpha_eff / (alpha_eff + beta_eff + 1e-300)
        elif bstr == "beta_ts":
            if rng is None: rng = np.random.default_rng()
            bandit_scores = rng.beta(alpha_eff, beta_eff)
        elif bstr == "beta_ucb":
            # Use same quantile scheduling inputs (quantile|delta|step_t, alpha)
            if quantile is not None:
                q_used_bandit = float(quantile)
            elif delta is not None:
                q_used_bandit = 1.0 - float(delta)
            else:
                t_sched = max(t_now, 2)
                q_used_bandit = 1.0 - 1.0 / (t_sched ** float(alpha))
            q_used_bandit = float(np.clip(q_used_bandit, 1e-6, 1.0 - 1e-6))
            z = _norm_ppf(q_used_bandit)

            mu = alpha_eff / (alpha_eff + beta_eff + 1e-300)
            var = (alpha_eff * beta_eff) / ((alpha_eff + beta_eff)**2 * (alpha_eff + beta_eff + 1.0) + 1e-300)
            sd = np.sqrt(np.maximum(var, 0.0))
            bandit_scores = np.clip(mu + z * sd, 0.0, 1.0)
        else:
            raise ValueError("bandit_strategy must be one of {'discounted_ucb','beta_ts','beta_ucb','beta_mean'}")

        idx_bandit = int(np.argmax(bandit_scores))

        bandit_out = {
            "index": idx_bandit,
            "strategy": bstr,
            "score": float(bandit_scores[idx_bandit]),
            "t": int(t_now),
            "discount": lam,
            "alpha0": float(bandit_alpha0),
            "beta0": float(bandit_beta0),
        }
        if bstr in {"beta_mean","beta_ts","beta_ucb"}:
            bandit_out.update({
                "alpha_eff": float(alpha_eff[idx_bandit]),
                "beta_eff": float(beta_eff[idx_bandit]),
            })
        else:
            bandit_out.update({
                "mean_hat": float((bandit_alpha0 + Seff[idx_bandit]) /
                                (bandit_alpha0 + bandit_beta0 + Neff[idx_bandit] + 1e-300)),
                "Neff": float(Neff[idx_bandit]),
            })
        if lik == "2pl":
            bandit_out["a"] = float(A[idx_bandit])
        else:
            bandit_out["ideal_kappa"] = float(K[idx_bandit])
            bandit_out["ideal_gamma"] = float(Gm[idx_bandit])

        # ======================= (C) CHOOSE ENGINE =======================
        eng = engine.lower()
        if eng not in {"bayes", "bandit", "hybrid"}:
            raise ValueError("engine must be one of {'bayes','bandit','hybrid'}")

        if eng == "bayes":
            chosen_idx = idx_bayes
        elif eng == "bandit":
            chosen_idx = idx_bandit
        else:
            # min-max normalize then mix
            def norm01(x):
                x = np.asarray(x, dtype=float)
                lo, hi = float(np.min(x)), float(np.max(x))
                return (x - lo) / (hi - lo + 1e-12)
            mix = (1.0 - float(hybrid_eta)) * norm01(expected_success) + float(hybrid_eta) * norm01(bandit_scores)
            chosen_idx = int(np.argmax(mix))

        # ======================= (D) OUTPUT =======================
        out = {
            "engine": eng,
            "index": int(chosen_idx),
            "b": float(B[chosen_idx]),
            "bayes": {
                "index": idx_bayes,
                "policy": policy,
                "expected_success": float(expected_success[idx_bayes]),
            },
            "bandit": bandit_out,
        }
        if lik == "2pl":
            out["a"] = float((A if A is not None else np.full_like(B, self.default_a))[chosen_idx])
        else:
            out["ideal_kappa"] = float(K[chosen_idx]); out["ideal_gamma"] = float(Gm[chosen_idx])

        if return_all:
            out["bayes_expected_all"] = expected_success
            out["bayes_scores_all"] = bayes_scores
            out["bandit_scores_all"] = bandit_scores
            if bstr in {"beta_mean","beta_ts","beta_ucb"}:
                out["bandit_alpha_all"] = alpha_eff
                out["bandit_beta_all"]  = beta_eff
        return out

    #------  State ----
    def state(self) -> Dict[str, Any]:
        """Snapshot of current posterior and history."""
        grid, post = self.posterior()
        return {
            "grid": grid.tolist(),
            "posterior": post.tolist(),
            "history": self.history.copy(),
            "theta_mean": self.estimate(method="mean")["theta"],
            "theta_map": self.estimate(method="map")["theta"],
        }

    # ------------ (Optional) 2PL item-calibration MMLE (unchanged model) --------
    def estimate_item_params(
        self,
        Y: Sequence[int | bool],
        *,
        init_a: Optional[float] = None,
        init_b: Optional[float] = None,
        a_bounds: Tuple[float, float] = (0.25, 20.0),
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = False,
        return_history: bool = False,
    ) -> Dict[str, Any]:
        """
        MMLE for a single 2PL item (integrating out θ under the prior).
        Optimizes (a,b) via gradient ascent in unconstrained params:
            a = exp(log_a) > 0,  b = sigmoid(logit_b) in (0,1).
        """
        import numpy as np

        # -------------------- validate data --------------------
        y = np.asarray(Y, dtype=int).ravel()
        if y.size == 0:
            raise ValueError("Y must be non-empty")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Y must be 0/1")
        N = int(y.size); n1 = int(y.sum()); n0 = N - n1

        # -------------------- grid + prior (log) --------------------
        grid = self.grid; G = grid.size
        delta = 1.0 / (G - 1)
        trap_w = np.full(G, delta, dtype=float); trap_w[0] *= 0.5; trap_w[-1] *= 0.5

        log_prior = self.log_prior.astype(float)
        log_prior -= np.max(log_prior)  # safe shift
        log_base = np.log(trap_w) + log_prior

        eps = float(self.eps)

        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        def logsumexp(v):
            m = float(np.max(v))
            return m + float(np.log(np.sum(np.exp(v - m))))

        # -------------------- initialization --------------------
        if init_b is None:
            prior_unnorm = np.exp(log_prior)
            Zp = float(np.sum(prior_unnorm * trap_w))
            prior_mean = float(np.sum(grid * prior_unnorm * trap_w) / (Zp + 1e-300))
            b0 = np.clip(prior_mean, 1e-6, 1.0 - 1e-6)
        else:
            b0 = float(np.clip(init_b, 1e-6, 1.0 - 1e-6))

        a0 = float(np.clip(self.default_a if init_a is None else init_a, a_bounds[0], a_bounds[1]))

        # Unconstrained params
        log_a  = float(np.log(a0))                 # a = exp(log_a)
        logit_b = float(np.log(b0 / (1.0 - b0)))   # b = sigmoid(logit_b)

        # -------------------- objective + gradient --------------------
        def objective_and_grad(log_a: float, logit_b: float):
            """Return ell, d_log_a, d_logit_b, a, b, expected_correct_rate_prior."""
            a = float(np.exp(log_a))
            b = float(sigmoid(logit_b))

            logits = np.clip(a * (grid - b), -50.0, 50.0)
            p = np.clip(sigmoid(logits), eps, 1.0 - eps)
            logp = np.log(p); log1mp = np.log1p(-p)

            # log m1 = log ∑ ω π p ; log m0 = log ∑ ω π (1-p)
            log_w1 = log_base + logp
            log_w0 = log_base + log1mp
            log_m1 = logsumexp(log_w1)
            log_m0 = logsumexp(log_w0)

            # marginal log-likelihood
            ell = n1 * log_m1 + n0 * log_m0

            # responsibilities
            r1 = np.exp(log_w1 - log_m1)  # sums to 1
            r0 = np.exp(log_w0 - log_m0)  # sums to 1

            # gradients in (a, b)
            term = (grid - b)
            dL_da = n1 * float(np.sum(r1 * (1.0 - p) * term)) + \
                    n0 * float(np.sum(r0 * (-p) * term))
            dL_db = (-a) * (n1 * float(np.sum(r1 * (1.0 - p))) +
                            n0 * float(np.sum(r0 * (-p))))

            # chain rule to unconstrained params
            d_log_a   = a * dL_da
            d_logit_b = b * (1.0 - b) * dL_db

            # diagnostic
            log_mp = logsumexp(log_base)  # log ∑ ω π
            expected_correct_rate_prior = float(np.exp(log_m1 - log_mp))

            return (ell, d_log_a, d_logit_b, a, b, expected_correct_rate_prior)

        # Evaluate at start
        ell, d_log_a, d_logit_b, a, b, ecp = objective_and_grad(log_a, logit_b)
        history = []
        if return_history:
            history.append({"iter": 0, "a": a, "b": b, "ell": ell,
                            "grad_norm": float(np.hypot(d_log_a, d_logit_b))})

        # -------------------- gradient ascent with backtracking --------------------
        converged = False
        for it in range(1, max_iter + 1):
            gvec = np.array([d_log_a, d_logit_b], dtype=float)
            gnorm = float(np.linalg.norm(gvec))
            if gnorm < tol:
                converged = True
                break

            # ascent direction (normalized gradient)
            dvec = gvec / (gnorm + 1e-12)

            # Armijo backtracking line search
            step = 1.0
            c = 1e-4
            old_ell = ell
            while True:
                log_a_new  = log_a  + step * dvec[0]
                logit_b_new = logit_b + step * dvec[1]

                # clamp a via log_a bounds
                a_new = float(np.exp(log_a_new))
                if a_new < a_bounds[0]:
                    log_a_new = float(np.log(a_bounds[0]))
                elif a_new > a_bounds[1]:
                    log_a_new = float(np.log(a_bounds[1]))

                ell_new, d_log_a_new, d_logit_b_new, a_tmp, b_tmp, ecp_new = objective_and_grad(log_a_new, logit_b_new)

                # Armijo condition
                if ell_new >= old_ell + c * step * gnorm:
                    break
                step *= 0.5
                if step < 1e-8:
                    # accept small improvement; prevents stalling
                    break

            # Accept step
            log_a, logit_b = log_a_new, logit_b_new
            ell, d_log_a, d_logit_b, a, b, ecp = ell_new, d_log_a_new, d_logit_b_new, a_tmp, b_tmp, ecp_new

            if return_history:
                history.append({"iter": it, "a": a, "b": b, "ell": ell, "step": step,
                                "grad_norm": float(np.hypot(d_log_a, d_logit_b))})

            if verbose and (it == 1 or it % 10 == 0):
                print(f"[iter {it:3d}] ell={ell:.6f}  a={a:.4f}  b={b:.4f}  "
                    f"|g|={np.hypot(d_log_a,d_logit_b):.3e}  step={step:.3f}")

            # Relative improvement check
            if abs(ell - old_ell) < tol * (1.0 + abs(old_ell)):
                converged = True
                break

        out = {
            "a": float(a),
            "b": float(b),
            "loglike": float(ell),
            "counts": {"n": int(N), "n_correct": int(n1)},
            "expected_correct_rate_prior": float(ecp),
            "converged": bool(converged),
            "n_iter": int(it if 'it' in locals() else 0),
            "grad_norm": float(np.hypot(d_log_a, d_logit_b)),
        }
        if return_history:
            out["history"] = history
        return out


class ItemizedBayesian2PL(Bayesian2PL):
    def next_item_from_dicts(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        likelihood: str = "2pl",
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level wrapper: given a list of item dictionaries, returns the best next item.
        """
        b_list, labels, extras = self._extract_item_params(items, likelihood=likelihood)
        result = self.next_item(
            candidates_b=b_list,
            candidates_a=extras.get("a"),
            ideal_kappa=extras.get("kappa"),
            ideal_gamma=extras.get("gamma"),
            likelihood=likelihood,
            **kwargs
        )
        result["item"] = self._attach_item_label(items, result["index"])
        return result

    def _extract_item_params(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        likelihood: str
    ) -> Tuple[List[float], List[str], Dict[str, List[Optional[float]]]]:
        b_list = []
        label_list = []
        a_list = []
        kappa_list = []
        gamma_list = []

        for item in items:
            if "label" not in item or "b" not in item:
                raise ValueError("Each item must have at least 'label' and 'b'")

            label_list.append(item["label"])
            b_list.append(float(item["b"]))

            if likelihood == "2pl":
                a_list.append(item.get("a"))
            elif likelihood == "ideal":
                kappa_list.append(item.get("kappa"))
                gamma_list.append(item.get("gamma", 0.0))
            else:
                raise ValueError("Unknown likelihood: must be '2pl' or 'ideal'")

        extras = {}
        if likelihood == "2pl":
            extras["a"] = a_list
        elif likelihood == "ideal":
            extras["kappa"] = kappa_list
            extras["gamma"] = gamma_list

        return b_list, label_list, extras

    def _attach_item_label(self, items: Sequence[Dict[str, Any]], index: int) -> Dict[str, Any]:
        if not (0 <= index < len(items)):
            raise IndexError(f"Invalid item index {index}")
        return items[index].copy()

class ItemResponsePredictionRunner:
    def __init__(self, *, grid_points: int = 1001, alpha: float = 1.0, beta: float = 1.0,
                 default_a: float = 4.0, likelihood: str = "ideal", engine: str = "hybrid",
                 policy: str = "ucb", bandit_discount: float = 0.97, bandit_c: float = 0.50,
                 bandit_alpha0: float = 1.0, bandit_beta0: float = 1.0, hybrid_eta: float = 0.5,
                 max_exposures_per_item: int = 0, items: Optional[Sequence[dict]] = None,
                 B: Optional[Sequence[float]] = None, A: Optional[Sequence[float]] = None,
                 K: Optional[Sequence[float]] = None, Gm: Optional[Sequence[float]] = None,
                 estimator: Optional[Any] = None):
        np.set_printoptions(suppress=True, linewidth=200, threshold=100000)

        # Config
        self.likelihood = likelihood.lower()
        self.engine = engine.lower()
        self.policy = policy.lower()
        self.bandit_discount = float(bandit_discount)
        self.bandit_c = float(bandit_c)
        self.bandit_alpha0 = float(bandit_alpha0)
        self.bandit_beta0 = float(bandit_beta0)
        self.hybrid_eta = float(hybrid_eta)
        self.max_exposures_per_item = int(max_exposures_per_item)

        # Candidate pool
        self.item_dicts = list(items) if items is not None else None
        if self.item_dicts is not None:
            self.B = np.array([it["b"] for it in self.item_dicts], dtype=float)
            if self.likelihood == "2pl":
                # use constructor arg default_a (estimator not built yet)
                self.A = np.array([it.get("a", float(default_a)) for it in self.item_dicts], dtype=float)
                self.K = self.Gm = None
            else:
                self.K  = np.array([it.get("kappa", 50.0) for it in self.item_dicts], dtype=float)
                self.Gm = np.array([it.get("gamma", 0.0)  for it in self.item_dicts], dtype=float)
                self.A = None
        else:
            if B is None:
                B = np.linspace(0.05, 0.95, 10)
            self.B = np.asarray(B, float).ravel()
            if self.likelihood == "2pl":
                self.A = np.asarray(A if A is not None else np.full_like(self.B, default_a, dtype=float), float)
                self.K = self.Gm = None
            else:
                self.K  = np.asarray(K  if K  is not None else np.full_like(self.B, 50.0, dtype=float), float)
                self.Gm = np.asarray(Gm if Gm is not None else np.zeros_like(self.B, dtype=float), float)
                self.A = None

        self.exposures = np.zeros(self.B.size, dtype=int)
        self.step_idx = 0

        # Estimator
        self.est = estimator if estimator is not None else ItemizedBayesian2PL(
            grid_points=grid_points, alpha=alpha, beta=beta, default_a=default_a
        )


    # ----------------------- (Re)initialize / reset -----------------------

    def reset_estimator(
        self,
        *,
        grid_points: Optional[int] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        default_a: Optional[float] = None,
        clear_bandit: bool = True,
        clear_exposures: bool = True,
        items: Optional[Sequence[dict]] = None,
        likelihood: Optional[str] = None,
    ) -> dict:
        """
        Reset or rebuild the estimator and (optionally) replace the item pool.

        Behavior
        --------
        - If `items` is provided: performs a full restart —
        builds a fresh estimator (prior from `alpha`,`beta`), installs the new items,
        clears bandit state and exposures, and resets the step counter.
        - Otherwise: rebuilds only if `grid_points`/`default_a` change; else soft-resets
        the estimator to the prior. Bandit/exposures are cleared per flags.

        Also returns a Plotly figure (HTML string) visualizing per-item probability
        profiles over θ (un-normalized), limited to a manageable subset for readability.

        Parameters
        ----------
        grid_points : Optional[int]
            Grid resolution for θ ∈ [0,1]. If provided with no `items`, triggers rebuild.
        alpha, beta : float
            Beta prior hyperparameters used for (re)initializing the θ prior.
        default_a : Optional[float]
            Default discrimination for 2PL; triggers rebuild if provided.
        clear_bandit : bool
            When True, clears per-arm bandit stats and time index.
        clear_exposures : bool
            When True, zeros the per-item exposure counters.
        items : Optional[Sequence[dict]]
            New item pool. For likelihood="2pl" each item needs {"label","b","a"};
            for "ideal" each needs {"label","b","kappa","gamma"}.
            Providing `items` forces a full rebuild.
        likelihood : Optional[str]
            Override likelihood ("2pl" or "ideal") for validation/plotting.

        Returns
        -------
        dict
            API envelope with:
            - response.meta_data: details (estimator config, posterior summary, reward stats),
            - response.data.figure: Plotly HTML for item probability curves (if items present),
            - status: "success" or "error".
        """

        # ---------- helpers ----------
        def _err(msg: str) -> dict:
            message=f"Error: {msg}"
            meta = {"action": "reset_estimator", "message": message}
            return {
                "status": "error",
                "response": {
                    "meta_data": meta,
                    "data": json.dumps({"figure": "", "records": [meta]}),
                    "message": message,
                },
            }

        def _item_prob(theta: np.ndarray, item: dict, lik: str) -> np.ndarray:
            """Return p_i(theta) for an item under likelihood."""
            if lik == "2pl":
                a = float(item["a"])
                b = float(item["b"])
                logits = np.clip(a * (theta - b), -50.0, 50.0)
            else:  # ideal
                b = float(item["b"]); kappa = float(item["kappa"]); gamma = float(item["gamma"])
                logits = np.clip(gamma - kappa * (theta - b) ** 2, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-logits))

        # ---------- apply likelihood change first ----------
        if likelihood is not None:
            self.likelihood = likelihood.lower()

        # ---------- validate & stage new items if provided ----------
        staged = None
        if items is not None:
            item_dicts = list(items)
            req = {"2pl": {"b", "a"}, "ideal": {"b", "kappa", "gamma"}}.get(self.likelihood)
            if req is None:
                return _err(f"Unsupported likelihood type: {self.likelihood}")
            for i, it in enumerate(item_dicts):
                missing = req - set(it.keys())
                if missing:
                    return _err(f"Item at index {i} missing keys for '{self.likelihood}': {sorted(missing)}")

            B = np.array([it["b"] for it in item_dicts], dtype=float)
            if self.likelihood == "2pl":
                A  = np.array([it["a"] for it in item_dicts], dtype=float)
                K = Gm = None
            else:
                K  = np.array([it["kappa"] for it in item_dicts], dtype=float)
                Gm = np.array([it["gamma"] for it in item_dicts], dtype=float)
                A  = None
            staged = {"item_dicts": item_dicts, "B": B, "A": A, "K": K, "Gm": Gm}

        # ---------- meta envelope ----------
        meta = {
            "action": "reset_estimator",
            "need_rebuild": bool((grid_points is not None) or (default_a is not None) or (staged is not None)),
            "requested": {
                "grid_points": int(grid_points) if grid_points is not None else None,
                "alpha": float(alpha),
                "beta": float(beta),
                "default_a": float(default_a) if default_a is not None else None,
                "clear_bandit": bool(clear_bandit),
                "clear_exposures": bool(clear_exposures),
                "likelihood": self.likelihood,
                "items_provided": bool(staged is not None),
            },
        }

        try:
            # ---------- rebuild logic ----------
            if staged is not None:
                # Force full rebuild when items change
                gp = int(grid_points) if grid_points is not None else int(getattr(self.est, "grid_points", 1001))
                da = float(default_a)  if default_a  is not None else float(getattr(self.est, "default_a", 4.0))
                self.est = ItemizedBayesian2PL(grid_points=gp, alpha=alpha, beta=beta, default_a=da)
                meta["result"] = "rebuilt_from_items"

                # Install new pool + arrays
                self.item_dicts = staged["item_dicts"]
                self.B, self.A, self.K, self.Gm = staged["B"], staged["A"], staged["K"], staged["Gm"]

                # Clear bandit & exposures & step counter
                self.est._bandit_stats = {}
                self.est._bandit_t = 0
                self.exposures = np.zeros(len(self.item_dicts), dtype=int)
                self.step_idx = 0

            else:
                # No new items: rebuild if structural args changed, else soft reset
                if (grid_points is not None) or (default_a is not None):
                    gp = int(grid_points) if grid_points is not None else int(self.est.grid_points)
                    da = float(default_a)  if default_a  is not None else float(self.est.default_a)
                    self.est = ItemizedBayesian2PL(grid_points=gp, alpha=alpha, beta=beta, default_a=da)
                    meta["result"] = "rebuilt"
                else:
                    self.est.reset(alpha=alpha, beta=beta)
                    meta["result"] = "reset"

                # Optional clears
                if clear_bandit:
                    self.est._bandit_stats = {}
                    self.est._bandit_t = 0
                if clear_exposures and hasattr(self, "exposures") and isinstance(self.exposures, np.ndarray):
                    self.exposures[:] = 0
                self.step_idx = 0

            # ---------- posterior summary (fresh prior state) ----------
            grid, post = self.est.posterior()
            Z = float(np.trapezoid(post, grid))
            post_n = post / (Z + 1e-300)
            cdf = np.cumsum(post_n); cdf /= cdf[-1]
            theta_mean = float(np.trapezoid(grid * post_n, grid))
            theta_map  = float(grid[int(np.argmax(post_n))])
            ci_lo = float(np.interp(0.05, cdf, grid))
            ci_hi = float(np.interp(0.95, cdf, grid))

            stats = self.est.running_reward_stats()

            meta.update({
                "estimator": {"grid_points": int(self.est.grid_points), "default_a": float(self.est.default_a)},
                "posterior_summary": {"theta_mean": theta_mean, "theta_map": theta_map, "ci_90": (ci_lo, ci_hi)},
                "reward_stats": {"trials": int(stats["trials"]), "reward_sum": float(stats["reward_sum"]), "avg_reward": float(stats["avg_reward"])},
                "exposures_total": int(np.sum(self.exposures)) if hasattr(self, "exposures") else None,
                "message": f"Estimator {meta['result']} successfully",
            })

            # ---------- FIGURE: item probability profiles (area-normalized) ----------
            fig_return = ""  # default: no figure
            if getattr(self, "item_dicts", None):
                theta = self.est.grid  # match estimator grid for clarity
                eps = 1e-300
                max_items_to_plot = 50  # avoid overcrowding
                items_to_plot = self.item_dicts[:max_items_to_plot]

                fig = go.Figure()
                n_plotted = 0
                for it in items_to_plot:
                    p = _item_prob(theta, it, self.likelihood)
                    area = float(np.trapezoid(p, theta))
                    if not np.isfinite(area) or area <= 0:
                        continue
                    p_norm = p #/ (area + eps)
                    lbl = it.get("label", f"b={it['b']:.2f}")
                    fig.add_trace(go.Scatter(
                        x=theta, y=p_norm, mode="lines", name=lbl,
                        hovertemplate="θ=%{x:.3f}<br>p_norm=%{y:.4f}<extra>"+lbl+"</extra>"
                    ))
                    n_plotted += 1

                # decorate
                title = "Item probability profiles (un-normalized)"
                fig.update_layout(
                    title=title,
                    xaxis_title="θ",
                    yaxis_title="p_i(θ)",  #/ ∫ p_i(θ) dθ",
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )

                plot_html = pio.to_html(
                    fig, full_html=False, config={"displaylogo": False, "responsive": True}, include_plotlyjs=True
                )
                fig_id = str(uuid.uuid4())[:8]
                fig_return = plot_html.replace("<div>", f'<div id="{fig_id}">', 1)

                # attach a small note in meta
                meta["item_plot"] = {
                    "normalization": "area",
                    "n_items_plotted": int(n_plotted),
                    "max_items_to_plot": int(max_items_to_plot),
                }

            # ---------- return ----------
            return {
                "status": "success",
                "response": {
                    "meta_data": meta,
                    "data": json.dumps({"figure": fig_return, "records": [meta]}),
                    "message": json.dumps(meta),
                },
                "message": json.dumps(meta),
            }

        except Exception as e:
            return _err(str(e))



    # ----------------------------- Public: recommend (optional) -----------------------------

    def _predictive_for_index(self, idx: int) -> float:
        """
        Posterior-predictive success for pool item idx, using the session's
        likelihood and current Bayesian2PL posterior.
        """
        b = float(self.B[idx])
        if self.likelihood == "2pl":
            a = float(self.A[idx])
            return self.est.predictive_prob_for(b=b, a=a, likelihood="2pl")
        else:
            k = float(self.K[idx]); g = float(self.Gm[idx])
            return self.est.predictive_prob_for(b=b, likelihood="ideal", kappa=k, gamma=g)


    def recommend(self) -> dict:
        """
        Select the next item to present **without modifying state**.

        Behavior
        --------
        Uses the configured picker to score the current candidate pool and returns a
        recommendation. Supports both:
        • itemized pool (`self.item_dicts`: list of dicts with "label" & params), and
        • legacy arrays (B[, A|K|Gm]).
        If `max_exposures_per_item > 0`, items at that cap are temporarily masked.

        Picker
        ------
        engine ∈ {"bayes","bandit","hybrid"}
        - "bayes": CAT over θ-posterior with policy ∈ {"thompson","ucb","greedy",
            "bayesucb","max_info","closest_b"}.
        - "bandit": discounted UCB / Beta variants over arm-level rewards.
        - "hybrid": convex mix of bayes and bandit scores (weight = `hybrid_eta`).

        Side effects
        ------------
        None. This does **not** update the θ-posterior, bandit stats, or exposures.
        Call `step(...)` later to commit an observed outcome.

        Returns
        -------
        dict
            API envelope with:
            - response.meta_data:
                step                : planned step number (current step + 1)
                idx                 : chosen pool index
                item                : full item dict (if available; otherwise synthesized)
                p_pred_before       : posterior-predictive success for the choice (pre-update)
                exposures           : current exposure count for the choice
                avg_reward, reward_sum, trials : running reward stats (global)
                picker_engine       : engine actually used
                bayes_index, bayes_item   : head-specific best (if available)
                bandit_index, bandit_item : head-specific best (if available)
                message             : human-readable summary
            - response.data.figure : "" (recommendation emits no figure)
            - status              : "success" or "error" (with error message)
            - message              : "A text message summary"
        """
        meta = {
            "step": int(getattr(self, "step_idx", 0) + 1),
            "likelihood": getattr(self, "likelihood", None),
            "engine": getattr(self, "engine", None),
            "policy": getattr(self, "policy", None),
            "bandit_discount": float(getattr(self, "bandit_discount", 0.97)),
        }

        try:
            # ---------------- Itemized version ----------------
            if self.item_dicts is not None:
                if self.max_exposures_per_item > 0:
                    allowed = (self.exposures < self.max_exposures_per_item)
                    if not np.any(allowed):
                        allowed = np.ones_like(self.exposures, dtype=bool)

                    items_filtered = [item for i, item in enumerate(self.item_dicts) if allowed[i]]
                    if not items_filtered:
                        raise ValueError("No available items to recommend after exposure filtering.")

                    pick = self.est.next_item_from_dicts(
                        items_filtered,
                        likelihood=self.likelihood,
                        engine=self.engine,
                        bayes_policy=self.policy,
                        bandit_discount=self.bandit_discount,
                        bandit_c=self.bandit_c,
                        bandit_alpha0=self.bandit_alpha0,
                        bandit_beta0=self.bandit_beta0,
                        hybrid_eta=self.hybrid_eta
                    )
                    idx_filtered = int(pick["index"])
                    idx = int(np.flatnonzero(allowed)[idx_filtered])
                else:
                    pick = self.est.next_item_from_dicts(
                        self.item_dicts,
                        likelihood=self.likelihood,
                        engine=self.engine,
                        bayes_policy=self.policy,
                        bandit_discount=self.bandit_discount,
                        bandit_c=self.bandit_c,
                        bandit_alpha0=self.bandit_alpha0,
                        bandit_beta0=self.bandit_beta0,
                        hybrid_eta=self.hybrid_eta
                    )
                    idx = int(pick["index"])

            # ---------------- Legacy array version ----------------
            else:
                if self.max_exposures_per_item > 0:
                    allowed = (self.exposures < self.max_exposures_per_item)
                    if not np.any(allowed):
                        allowed = np.ones_like(self.exposures, dtype=bool)

                    B_f = self.B[allowed]
                    A_f = self.A[allowed] if self.likelihood == "2pl" else None
                    K_f = self.K[allowed] if self.likelihood == "ideal" else None
                    Gm_f = self.Gm[allowed] if self.likelihood == "ideal" else None

                    if B_f.size == 0:
                        raise ValueError("No available items to recommend after exposure filtering.")

                    pick = self.est.next_item(
                        candidates_b=B_f,
                        candidates_a=A_f,
                        ideal_kappa=K_f,
                        ideal_gamma=Gm_f,
                        likelihood=self.likelihood,
                        engine=self.engine,
                        bayes_policy=self.policy,
                        bandit_discount=self.bandit_discount,
                        bandit_c=self.bandit_c,
                        bandit_alpha0=self.bandit_alpha0,
                        bandit_beta0=self.bandit_beta0,
                        hybrid_eta=self.hybrid_eta,
                        return_all=False,
                    )
                    idx_filtered = int(pick["index"])
                    idx = int(np.flatnonzero(allowed)[idx_filtered])
                else:
                    pick = self.est.next_item(
                        candidates_b=self.B,
                        candidates_a=self.A if self.likelihood == "2pl" else None,
                        ideal_kappa=self.K if self.likelihood == "ideal" else None,
                        ideal_gamma=self.Gm if self.likelihood == "ideal" else None,
                        likelihood=self.likelihood,
                        engine=self.engine,
                        bayes_policy=self.policy,
                        bandit_discount=self.bandit_discount,
                        bandit_c=self.bandit_c,
                        bandit_alpha0=self.bandit_alpha0,
                        bandit_beta0=self.bandit_beta0,
                        hybrid_eta=self.hybrid_eta,
                        return_all=False,
                    )
                    idx = int(pick["index"])

            # ---------------- Predictive Probability ----------------
            b_i = float(self.B[idx])
            p_pred_before = self._predictive_for_index(idx)

            if self.likelihood == "2pl":
                knobs = {"a": float(self.A[idx])}
            else:
                knobs = {"kappa": float(self.K[idx]), "gamma": float(self.Gm[idx])}

            exp = int(self.exposures[idx])
            stats = self.est.running_reward_stats()

            # Build full item dict
            item = self.item_dicts[idx].copy() if self.item_dicts is not None else {
                "b": b_i,
                "a": float(self.A[idx]) if self.likelihood == "2pl" else None,
                "kappa": float(self.K[idx]) if self.likelihood == "ideal" else None,
                "gamma": float(self.Gm[idx]) if self.likelihood == "ideal" else None,
                "label": f"Item-{idx}"
            }
            # Remove None values from the item dict
            item = {k: v for k, v in item.items() if v is not None}
            message = f"Recommended item is {item}"
            meta.update({
                "idx": idx,
                "item": item,
                "p_pred_before": float(p_pred_before),
                "exposures": exp,
                "avg_reward": float(stats["avg_reward"]),
                "reward_sum": float(stats["reward_sum"]),
                "trials": int(stats["trials"]),
                "picker_engine": pick.get("engine", self.engine) if isinstance(pick, dict) else self.engine,
                "bayes_index": None,
                "bayes_item": None,
                "bandit_index": None,
                "bandit_item": None,
                "message": message,
            })

            if isinstance(pick, dict):
                bayes_idx = pick.get("bayes", {}).get("index")
                bandit_idx = pick.get("bandit", {}).get("index")

                if bayes_idx is not None:
                    meta["bayes_index"] = int(bayes_idx)
                    if self.item_dicts is not None:
                        meta["bayes_item"] = self.item_dicts[int(bayes_idx)].copy()

                if bandit_idx is not None:
                    meta["bandit_index"] = int(bandit_idx)
                    if self.item_dicts is not None:
                        meta["bandit_item"] = self.item_dicts[int(bandit_idx)].copy()



            # Attach full item dict if available
            if self.item_dicts is not None:
                meta["item"] = self.item_dicts[idx].copy()

            
            return {
                "status": "success",
                "response": {
                    "meta_data": meta,
                    "data": json.dumps({"figure": "", "records": [meta]}),
                    "message": message,
                },
                "message": message,
            }

        except Exception as e:
            message = f"Error: {str(e)}"
            meta["message"] = message
            return {
                "status": "error",
                "response": {
                    "meta_data": meta,
                    "data": json.dumps({"figure": "", "records": [meta]}),
                    "message": message,
                },
                "message": message,
            }


    # ----------------------------- Plot helpers -----------------------------

    @staticmethod
    def _plot_theta_posterior(grid, post, *, step=None, show=True, save_html=None, return_fig=False):
        grid = np.asarray(grid, float).ravel()
        post = np.asarray(post, float).ravel()
        Z = float(np.trapezoid(post, grid))
        post_n = post / (Z + 1e-300)
        mean_theta = float(np.trapezoid(grid * post_n, grid))
        map_theta  = float(grid[np.argmax(post_n)])

        title = "Theta posterior" + (f" (step {int(step)})" if step is not None else "")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grid, y=post_n, mode="lines", name="posterior",
            hovertemplate="theta=%{x:.3f}<br>density=%{y:.4f}<extra></extra>"
        ))
        fig.add_vline(x=mean_theta, line_width=2, line_dash="dash", line_color="black",
                      annotation_text=f"mean={mean_theta:.3f}", annotation_position="top left")
        fig.add_vline(x=map_theta, line_width=2, line_dash="dot", line_color="black",
                      annotation_text=f"map={map_theta:.3f}", annotation_position="top right")
        fig.update_layout(
            title=title, xaxis_title="theta", yaxis_title="density",
            template="plotly_white", margin=dict(l=60, r=20, t=60, b=50), hovermode="x unified"
        )
        if save_html:
            fig.write_html(save_html, include_plotlyjs="cdn", auto_open=False)
        if show:
            fig.show()
        if return_fig:
            return fig

    def _plot_b_success_profile(self, *, a=None, kappa=None, gamma=0.0,
                                num_points=400, show=True, save_html=None, return_fig=False):
        grid, post = self.est.posterior()
        b_grid = np.linspace(0.0, 1.0, num_points)

        if self.likelihood == "2pl":
            a_val = self.est.default_a if a is None else float(a)
            logits = np.clip(a_val * (grid[:, None] - b_grid[None, :]), -50.0, 50.0)
            title = f"Posterior-implied success vs b (2PL, a={a_val:.3f})"
        else:
            k_val = 50.0 if kappa is None else float(kappa)
            logits = np.clip(gamma - k_val * (grid[:, None] - b_grid[None, :])**2, -50.0, 50.0)
            title = f"Posterior-implied success vs b (ideal, kappa={k_val:.1f}, gamma={gamma:.1f})"

        p = 1.0 / (1.0 + np.exp(-logits))                 # (G, B)
        curve = np.trapezoid(p * post[:, None], grid, axis=0) # (B,)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=b_grid, y=curve, mode="lines", name="E[success | b]",
            hovertemplate="b=%{x:.3f}<br>P=%{y:.3f}<extra></extra>"
        ))

        # mark current pool b's
        Bs = np.array(self.B, float)
        if self.likelihood == "2pl":
            logits_pts = np.clip((self.est.default_a if a is None else float(a)) * (grid[:, None] - Bs[None, :]), -50.0, 50.0)
        else:
            kval = 50.0 if kappa is None else float(kappa)
            logits_pts = np.clip(gamma - kval * (grid[:, None] - Bs[None, :])**2, -50.0, 50.0)
        p_pts = 1.0 / (1.0 + np.exp(-logits_pts))
        curve_pts = np.trapezoid(p_pts * post[:, None], grid, axis=0)
        fig.add_trace(go.Scatter(
            x=Bs, y=curve_pts, mode="markers", name="pool b",
            marker=dict(size=8, symbol="circle-open"),
            hovertemplate="b=%{x:.3f}<br>P_pool=%{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=title, xaxis_title="b", yaxis_title="posterior-expected success",
            template="plotly_white", margin=dict(l=60, r=20, t=60, b=50), hovermode="x unified"
        )
        if save_html:
            fig.write_html(save_html, include_plotlyjs="cdn", auto_open=False)
        if show:
            fig.show()
        if return_fig:
            return fig

    @staticmethod
    def _print_posterior_details(grid, post, ci_level=0.90):
        grid = np.asarray(grid, dtype=float).ravel()
        post = np.asarray(post, dtype=float).ravel()
        Z = float(np.trapezoid(post, grid))
        norm_post = post / (Z + 1e-300)
        cdf = np.cumsum(norm_post); cdf /= cdf[-1]
        mean = float(np.trapezoid(grid * norm_post, grid))
        median = float(np.interp(0.5, cdf, grid))
        theta_map = float(grid[np.argmax(norm_post)])
        lo = float(np.interp((1.0 - ci_level)/2.0, cdf, grid))
        hi = float(np.interp(1.0 - (1.0 - ci_level)/2.0, cdf, grid))
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -float(np.trapezoid(norm_post * np.log(norm_post + 1e-300), grid))
        print(
            f"\n--- Posterior theta details ---\n"
            f"grid_size={grid.size}, normalized_Z≈{Z:.6f}\n"
            f"mean={mean:.4f}, median={median:.4f}, map={theta_map:.4f}\n"
            f"{int(ci_level*100)}% CI=({lo:.4f}, {hi:.4f}), entropy={ent:.4f}"
        )

    # ------------------------------- Step --------------------------------
    # NOTE: now step TAKES the chosen item index and the observed outcome y.

    def step(self, label: str, y: int, *, show_plots: bool = False):
        """
        Commit a single observed response for the item identified by `label`.

        Behavior
        --------
        - Looks up the item in `self.item_dicts` by its "label".
        - Updates the θ-posterior using the chosen likelihood ("2pl" or "ideal") with the raw
        success probability p(θ|item), and records the outcome into the bandit head.
        - Increments per-item exposure and the global step counter.
        - Returns a Plotly figure (HTML) of the updated θ posterior and rich metadata.

        Parameters
        ----------
        label : str
            Unique item label present in each item dict (e.g., "Q1", "item_3").
        y : int
            Binary outcome in {0, 1}.
        show_plots : bool, default False
            If True, displays the Plotly figure inline; the HTML is always returned.

        Returns
        -------
        dict
            API envelope with:
            - response.meta_data:
                step, item (full dict), y
                p_pred_before      : posterior-predictive success prior to the update
                theta_hat          : current θ point estimate (per `estimate(..., method="mean")`)
                ci_lo, ci_hi       : equal-tailed 90% credible interval bounds
                avg_reward, reward_sum, trials : running reward statistics (global)
                exposures_before / exposures_after
                message            : human-readable summary
            - response.data.figure : Plotly HTML of the updated θ posterior
            - status               : "success" or "error" (with message)
        """
        try:
            if y not in (0, 1):
                raise ValueError("y must be 0 or 1")

            # Lookup item and index by label
            if self.item_dicts is None:
                raise RuntimeError("item_dicts is not available — cannot use label-based step()")

            label_to_index = {item["label"]: i for i, item in enumerate(self.item_dicts)}
            if label not in label_to_index:
                raise ValueError(f"Label '{label}' not found in item_dicts")

            idx = label_to_index[label]
            item = self.item_dicts[idx]
            b_i = float(item["b"])
            exposures_before = int(self.exposures[idx])
            p_pred_before = self._predictive_for_index(idx)

            # Likelihood-specific
            if self.likelihood == "2pl":
                a_i = float(item["a"])
                out = self.est.estimate(
                    Y=[y], b=[b_i], a=[a_i],
                    likelihood="2pl", method="mean", ci_level=0.90, return_posterior=True
                )
                self.est.bandit_record_outcome(
                    b=b_i, a=a_i, y=y, likelihood="2pl", discount=self.bandit_discount
                )
                knob = {"a": a_i}
            else:
                k_i = float(item["kappa"])
                g_i = float(item["gamma"])
                out = self.est.estimate(
                    Y=[y], b=[b_i], kappa=[k_i], gamma=[g_i],
                    likelihood="ideal", method="mean", ci_level=0.90, return_posterior=True
                )
                self.est.bandit_record_outcome(
                    b=b_i, kappa=k_i, gamma=g_i, y=y, likelihood="ideal", discount=self.bandit_discount
                )
                knob = {"kappa": k_i, "gamma": g_i}

            # Bookkeeping
            self.exposures[idx] += 1
            self.step_idx += 1

            # Posterior summary
            grid = np.asarray(out["grid"], float).ravel()
            post = np.asarray(out["posterior"], float).ravel()
            Z = float(np.trapezoid(post, grid))
            post_n = post / (Z + 1e-300)
            cdf = np.cumsum(post_n); cdf /= cdf[-1]
            theta_mean = float(np.trapezoid(grid * post_n, grid))
            theta_map  = float(grid[np.argmax(post_n)])
            ci_lo = float(np.interp(0.05, cdf, grid))
            ci_hi = float(np.interp(0.95, cdf, grid))

            # Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grid, y=post_n, mode="lines", name="posterior",
                hovertemplate="θ=%{x:.3f}<br>density=%{y:.4f}<extra></extra>"
            ))
            fig.add_vline(x=theta_mean, line_dash="dash", line_color="black")
            fig.add_vline(x=theta_map,  line_dash="dot",  line_color="black")
            fig.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor="LightBlue", opacity=0.2, layer="below", line_width=0)
            fig.add_annotation(x=theta_mean, y=1.02, yref="paper", showarrow=False, text=f"mean={theta_mean:.3f}")
            fig.add_annotation(x=theta_map,  y=1.02, yref="paper", showarrow=False, text=f"map={theta_map:.3f}")
            fig.add_annotation(x=(ci_lo+ci_hi)/2.0, y=1.08, yref="paper", showarrow=False, text="90% CI")
            fig.update_layout(
                title=f"Theta posterior (step {self.step_idx})",
                xaxis_title="theta",
                yaxis_title="density",
                template="plotly_white",
                margin=dict(l=60, r=20, t=60, b=50),
                hovermode="x unified"
            )

            if show_plots:
                fig.show()

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
                include_plotlyjs=True,
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace("<div>", f'<div id="{fig_id}">', 1)

            # Assemble meta
            stats = self.est.running_reward_stats()
            message = f"Theta posterior updated from {item} response outcome = {y}"
            meta = {
                "step": self.step_idx,
                "y": y,
                "item": item.copy(),
                **knob,
                "p_pred_before": float(p_pred_before),
                "theta_hat": float(out["theta"]),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "avg_reward": float(stats["avg_reward"]),
                "reward_sum": float(stats["reward_sum"]),
                "trials": int(stats["trials"]),
                "exposures_before": exposures_before,
                "exposures_after": int(self.exposures[idx]),
                "message": message,
            }

            return {
                "status": "success",
                "response": {
                    "meta_data": meta,
                    "data": json.dumps({"figure": fig_return, "records": [meta]}),
                    "message": message,
                },
            }

        except Exception as e:
            meta = {
                "step": getattr(self, "step_idx", 0) + 1,
                "label": label,
                "message": f"Error: {str(e)}",
            }
            return {
                "status": "error",
                "response": {
                    "meta_data": meta,
                    "data": json.dumps({"figure": "", "records": [meta]}),
                    "message": json.dumps(meta),
                },
            }


class ItemResponsePrediction:
    def __init__(
        self,
        *,
        # Bayesian2PL init
        grid_points: int = 1001,
        alpha: float = 1.0,
        beta: float = 1.0,
        default_a: float = 4.0,
        # Selection config (used by recommend())
        likelihood: str = "ideal",            # "2pl" | "ideal"
        engine: str = "hybrid",               # "bayes" | "bandit" | "hybrid"
        policy: str = "ucb",                  # "thompson" | "greedy" | "ucb" | "bayesucb" | "max_info" | "closest_b"
        bandit_discount: float = 0.97,
        bandit_c: float = 0.50,
        bandit_alpha0: float = 1.0,
        bandit_beta0: float = 1.0,
        hybrid_eta: float = 0.5,
        max_exposures_per_item: int = 0,      # 0 = unlimited repeats
        # Candidate pool
        items: Optional[Sequence[dict]] = None,
        B: Optional[Sequence[float]] = None,
        A: Optional[Sequence[float]] = None,  # 2PL only
        K: Optional[Sequence[float]] = None,  # ideal only
        Gm: Optional[Sequence[float]] = None, # ideal only
        # External estimator (if already constructed)
        estimator: Optional[Any] = None,      # pass a Bayesian2PL if you already have one
    ):
        self.runner = ItemResponsePredictionRunner(
            grid_points=grid_points, alpha=alpha, beta=beta, default_a=default_a,
            likelihood=likelihood, engine=engine, policy=policy,
            bandit_discount=bandit_discount, bandit_c=bandit_c, bandit_alpha0=bandit_alpha0, bandit_beta0=bandit_beta0,
            hybrid_eta=hybrid_eta, max_exposures_per_item=max_exposures_per_item,
            items=items,
            B=B,
            K=K,
            Gm=Gm,
        )

    def recommend(self):
        """
        Select the next item to present **without modifying state**.

        Behavior
        --------
        Uses the configured picker to score the current candidate pool and returns a
        recommendation. Supports both:
        • itemized pool (`self.item_dicts`: list of dicts with "label" & params), and
        • legacy arrays (B[, A|K|Gm]).
        If `max_exposures_per_item > 0`, items at that cap are temporarily masked.

        Picker
        ------
        engine ∈ {"bayes","bandit","hybrid"}
        - "bayes": CAT over θ-posterior with policy ∈ {"thompson","ucb","greedy",
            "bayesucb","max_info","closest_b"}.
        - "bandit": discounted UCB / Beta variants over arm-level rewards.
        - "hybrid": convex mix of bayes and bandit scores (weight = `hybrid_eta`).

        Side effects
        ------------
        None. This does **not** update the θ-posterior, bandit stats, or exposures.
        Call `step(...)` later to commit an observed outcome.

        Returns
        -------
        dict
            API envelope with:
            - response.meta_data:
                step                : planned step number (current step + 1)
                idx                 : chosen pool index
                item                : full item dict (if available; otherwise synthesized)
                p_pred_before       : posterior-predictive success for the choice (pre-update)
                exposures           : current exposure count for the choice
                avg_reward, reward_sum, trials : running reward stats (global)
                picker_engine       : engine actually used
                bayes_index, bayes_item   : head-specific best (if available)
                bandit_index, bandit_item : head-specific best (if available)
                message             : human-readable summary
            - response.data.figure : "" (recommendation emits no figure)
            - status              : "success" or "error" (with error message)
        """

        return self.runner.recommend()

    def update_estimator(self, label: str, outcome: int, *, show_plots: bool = False):
        """
        Commit a single observed response for the item identified by `label`.

        Behavior
        --------
        - Looks up the item in `self.item_dicts` by its "label".
        - Updates the θ-posterior using the chosen likelihood ("2pl" or "ideal") with the raw
        success probability p(θ|item), and records the outcome into the bandit head.
        - Increments per-item exposure and the global step counter.
        - Returns a Plotly figure (HTML) of the updated θ posterior and rich metadata.

        Parameters
        ----------
        label : str
            Unique item label present in each item dict (e.g., "Q1", "item_3").
        y : int
            Binary outcome in {0, 1}.
        show_plots : bool, default False
            If True, displays the Plotly figure inline; the HTML is always returned.

        Returns
        -------
        dict
            API envelope with:
            - response.meta_data:
                step, item (full dict), y
                p_pred_before      : posterior-predictive success prior to the update
                theta_hat          : current θ point estimate (per `estimate(..., method="mean")`)
                ci_lo, ci_hi       : equal-tailed 90% credible interval bounds
                avg_reward, reward_sum, trials : running reward statistics (global)
                exposures_before / exposures_after
                message            : human-readable summary
            - response.data.figure : Plotly HTML of the updated θ posterior
            - status               : "success" or "error" (with message)
        """
        return self.runner.step(label=label, y=outcome, show_plots=show_plots)
    
    def reset_estimator(
        self,
        alpha: float = None, 
        beta: float = None, 
        clear_bandit: bool = True, 
        clear_exposures: bool = True,
        items: Optional[Sequence[dict]] = None,
        likelihood: Optional[str] = None
        ):
        """
        Reset or rebuild the estimator and (optionally) replace the item pool.

        Behavior
        --------
        - If `items` is provided: performs a full restart —
        builds a fresh estimator (prior from `alpha`,`beta`), installs the new items,
        clears bandit state and exposures, and resets the step counter.
        - Otherwise: rebuilds only if `grid_points`/`default_a` change; else soft-resets
        the estimator to the prior. Bandit/exposures are cleared per flags.

        Also returns a Plotly figure (HTML string) visualizing per-item probability
        profiles over θ (un-normalized), limited to a manageable subset for readability.

        Parameters
        ----------
        grid_points : Optional[int]
            Grid resolution for θ ∈ [0,1]. If provided with no `items`, triggers rebuild.
        alpha, beta : float
            Beta prior hyperparameters used for (re)initializing the θ prior.
        default_a : Optional[float]
            Default discrimination for 2PL; triggers rebuild if provided.
        clear_bandit : bool
            When True, clears per-arm bandit stats and time index.
        clear_exposures : bool
            When True, zeros the per-item exposure counters.
        items : Optional[Sequence[dict]]
            New item pool. For likelihood="2pl" each item needs {"label","b","a"};
            for "ideal" each needs {"label","b","kappa","gamma"}.
            Providing `items` forces a full rebuild.
        likelihood : Optional[str]
            Override likelihood ("2pl" or "ideal") for validation/plotting.

        Returns
        -------
        dict
            API envelope with:
            - response.meta_data: details (estimator config, posterior summary, reward stats),
            - response.data.figure: Plotly HTML for item probability curves (if items present),
            - status: "success" or "error".
        """
        return self.runner.reset_estimator(
            alpha=alpha, 
            beta=beta, 
            clear_bandit=clear_bandit, 
            clear_exposures=clear_exposures, 
            items=items,
            likelihood=likelihood
            )
