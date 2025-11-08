# ItemResponsePrediction

A high-level, production-friendly wrapper for adaptive item recommendation and Bayesian ability estimation over (\theta \in [0,1]), built on top of:

* `Bayesian2PL` — grid-based Bayesian estimator for 2PL and ideal-point likelihoods
* `ItemizedBayesian2PL` — convenience helpers for item dict pools
* `ItemResponsePredictionRunner` — orchestration (posterior, bandit head, plots)

`ItemResponsePrediction` exposes a clean, minimal API for:

* **Recommending the next item** to present based on your chosen engine/policy
* **Updating the posterior** and bandit statistics from an observed response
* **Resetting/replacing the item pool** (with optional plots)

---

## Features

* **Two likelihoods**

  * 2PL logistic (parameters (a>0), (b \in [0,1]))
  * Ideal-point logistic (parameters (\kappa>0), (\gamma), (b \in [0,1]))
* **Engines**: `bayes`, `bandit`, `hybrid`
* **Policies** (Bayes): `thompson`, `greedy`, `ucb`, `bayesucb`, `max_info`, `closest_b`
* **Bandit strategies** (via runner): discounted UCB and Beta variants
* **Posterior-predictive scoring** of items
* **Exposure caps** (temporarily mask over-exposed items)
* **Plotly visualizations** for posterior and per-item curves
* **Audit-friendly metadata** returned with each call

---

## Installation & Requirements

**Requirements**

* Python ≥ 3.9
* `numpy` (required)
* `plotly` (optional, only for plotting utilities)

**Install (from GitHub)**

```bash
pip install "git+https://github.com/mugalan/item-response-prediction.git"
```

**Optional: plotting support**

```bash
pip install plotly
```

> If your package defines an extra named `plotting`, you can alternatively do:
>
> ```bash
> pip install "git+https://github.com/mugalan/item-response-prediction.git#egg=item-response-prediction[plotting]"
> ```

**Import**

```python
from item_response_prediction import ItemResponsePrediction
```

**Colab tip**
After installing, run the import in a new cell (or `Runtime > Restart session`) so the environment picks up the newly installed package.

**Developer (editable) install**

```bash
git clone https://github.com/mugalan/item-response-prediction.git
cd item-response-prediction
pip install -e .
```
---

## Concepts at a Glance

* **Ability ((\theta))**: Learner/user latent skill; supported on `[0,1]` with a Beta prior.
* **Item parameters**

  * 2PL: difficulty `b ∈ [0,1]`, discrimination `a > 0`.
  * Ideal-point: location `b ∈ [0,1]`, sharpness `κ > 0`, height/offset `γ`.
* **Posterior**: Grid approximation (default 1001 points) with numerically safe log-space normalization.
* **Recommendation**: Combines Bayesian CAT-style scoring and/or bandit exploration; hybrid mixes both.

---

## Quick Start

```python
import json
from IPython.display import display, HTML
items = [
    {"label": "Q1", "b": 0.30, "kappa": 40.0, "gamma": 0.0},
    {"label": "Q2", "b": 0.45, "kappa": 50.0, "gamma": 0.0},
    {"label": "Q3", "b": 0.60, "kappa": 60.0, "gamma": 0.0},
]

irp = ItemResponsePrediction(
    likelihood="ideal",      # or "2pl"
    engine="hybrid",         # "bayes" | "bandit" | "hybrid"
    policy="ucb",            # used by bayes engine
    hybrid_eta=0.5,
    items=items,
)

# 1) Ask for a recommendation (does NOT modify state)
rec = irp.recommend()
print(rec["response"]["meta_data"]["item"])  # the recommended item dict

# 2) Record an observed outcome for that label (updates posterior & bandit)
label = rec["response"]["meta_data"]["item"]["label"]
step_out = irp.update_estimator(label=label, outcome=1)  # 1=correct/positive, 0=incorrect/negative
figure=HTML(json.loads(step_out['response']['data']).get('figure'))
display(figure)

# 3) Repeat recommend → update loop
```

---

## Data Schemas

### Item dictionaries (recommended)

* **2PL**: `{"label": str, "b": float in [0,1], "a": float > 0}`
* **Ideal-point**: `{"label": str, "b": float in [0,1], "kappa": float > 0, "gamma": float}`

### Legacy arrays (also supported by the runner)

* Provide `B` (and `A` or `K`/`Gm` depending on the likelihood). `ItemResponsePrediction` prefers item dicts.

---

## API Reference: `ItemResponsePrediction`

### Constructor

```python
ItemResponsePrediction(
    *,
    grid_points: int = 1001,
    alpha: float = 1.0,
    beta: float = 1.0,
    default_a: float = 4.0,
    likelihood: str = "ideal",
    engine: str = "hybrid",
    policy: str = "ucb",
    bandit_discount: float = 0.97,
    bandit_c: float = 0.50,
    bandit_alpha0: float = 1.0,
    bandit_beta0: float = 1.0,
    hybrid_eta: float = 0.5,
    max_exposures_per_item: int = 0,
    items: Optional[Sequence[dict]] = None,
    B: Optional[Sequence[float]] = None,
    A: Optional[Sequence[float]] = None,
    K: Optional[Sequence[float]] = None,
    Gm: Optional[Sequence[float]] = None,
    estimator: Optional[Any] = None,
)
```

* **`likelihood`**: `"2pl"` or `"ideal"`.
* **`engine`**: `"bayes"`, `"bandit"`, or `"hybrid"`.
* **`policy`** (Bayes engine): `"thompson" | "greedy" | "ucb" | "bayesucb" | "max_info" | "closest_b"`.
* **`max_exposures_per_item`**: `0` means unlimited; `>0` temporarily masks over-exposed items.

### `recommend()` → dict

Returns an envelope with `status`, `message`, and `response`:

```python
{
  "status": "success",
  "response": {
    "meta_data": {
      "step": int,
      "idx": int,
      "item": { ... },
      "p_pred_before": float,
      "exposures": int,
      "avg_reward": float,
      "reward_sum": float,
      "trials": int,
      "picker_engine": "bayes" | "bandit" | "hybrid",
      "bayes_index": Optional[int],
      "bandit_index": Optional[int],
      ...
    },
    "data": {"figure": ""},
    "message": str
  }
}
```

> Side-effect free: does not update posterior or bandit stats.

### `update_estimator(label: str, outcome: int, show_plots: bool=False)` → dict

* Updates the posterior with the observed binary outcome for the given item `label`.
* Records the outcome into the bandit head (discounted counts or Beta posterior depending on strategy).
* Returns an envelope including a Plotly HTML string of the updated (\theta) posterior.

### `reset_estimator(alpha=None, beta=None, clear_bandit=True, clear_exposures=True, items=None, likelihood=None)` → dict

* **Soft reset**: when only `alpha/beta` change (and no new items), the grid estimator is reset to the prior.
* **Full rebuild**: when `items` are provided; installs new pool, clears bandit state and exposures.
* Returns an envelope with metadata and (if items provided) a Plotly HTML figure showing per-item probability profiles.

---

## Engines & Policies (Behavioral Summary)

### Bayes engine

Scores each candidate by posterior-expected success (or related info):

* `greedy`: maximize expected success
* `ucb`: expected success + κ × posterior std
* `thompson`: sample (\theta) from the posterior; pick best given that draw
* `bayesucb`: quantile-based optimistic selection w.r.t. (\theta) or (p)
* `max_info`: maximize Fisher information under the posterior mixture
* `closest_b`: heuristic — pick item with `b` closest to current (E[\theta])

### Bandit engine

Tracks *arm-level* discounted rewards and chooses with:

* Discounted UCB
* Beta-TS / Beta-UCB / Beta-mean variants (effective α/β updated with discounting)

### Hybrid engine

Convex combination of normalized Bayes and Bandit scores: `mix = (1-η)*Bayes + η*Bandit`.

---

## Likelihood Details

### 2PL Logistic

[ p(y=1 \mid \theta, a, b) = \sigma\big(a(\theta - b)\big) ]

* `a > 0` (discrimination); `b ∈ [0,1]` (difficulty)

### Ideal-Point Logistic

[ p(y=1 \mid \theta, \kappa, \gamma, b) = \sigma\big(\gamma - \kappa (\theta - b)^2\big) ]

* `κ > 0` (sharpness); `γ` (height/offset); `b ∈ [0,1]`

Both are numerically stabilized by clipping logits and probabilities.

---

## End-to-End Examples

### A. Ideal-point, item dict pool

```python
import json
from IPython.display import display, HTML

items = [
    {"label": "I1", "b": 0.25, "kappa": 35.0, "gamma": 0.0},
    {"label": "I2", "b": 0.50, "kappa": 50.0, "gamma": 0.0},
    {"label": "I3", "b": 0.75, "kappa": 70.0, "gamma": 0.0},
]

irp = ItemResponsePrediction(likelihood="ideal", engine="hybrid", policy="ucb", items=items)

for t in range(5):
    rec = irp.recommend()
    label = rec["response"]["meta_data"]["item"]["label"]

    import random
    y = 1 if random.random() < (0.8 if label == "I2" else 0.5) else 0

    out = irp.update_estimator(label, y)
    fig_html = json.loads(out["response"]["data"])["figure"]
    display(HTML(fig_html))             
    print(out["response"]["meta_data"]["theta_hat"],
          out["response"]["meta_data"]["avg_reward"])
```

### B. 2PL, item dict pool with discrimination

```python
import json
from IPython.display import display, HTML
items_2pl = [
    {"label": "Q1", "b": 0.35, "a": 3.5},
    {"label": "Q2", "b": 0.50, "a": 4.0},
    {"label": "Q3", "b": 0.70, "a": 5.0},
]

irp2 = ItemResponsePrediction(likelihood="2pl", engine="bayes", policy="thompson", items=items_2pl)
rec = irp2.recommend()
print("Recommend:", rec["response"]["meta_data"]["item"])  # inspect choice

# Suppose user answered Q2 incorrectly (0)
out = irp2.update_estimator("Q2", 0)
fig_html = json.loads(out["response"]["data"])["figure"]
display(HTML(fig_html))             
print("theta(mean):", out["response"]["meta_data"]["theta_hat"])
```

### C. Reset with a new pool and prior

```python
new_items = [
    {"label": "N1", "b": 0.40, "kappa": 50.0, "gamma": 0.0},
    {"label": "N2", "b": 0.60, "kappa": 50.0, "gamma": 0.0},
]

r = irp.reset_estimator(alpha=2.0, beta=2.0, items=new_items, likelihood="ideal")
fig_html = json.loads(out["response"]["data"])["figure"]
display(HTML(fig_html))             # <-- actually display it
print(out["response"]["meta_data"]["theta_hat"],
      out["response"]["meta_data"]["avg_reward"])
print(r["status"], r["response"]["meta_data"]["posterior_summary"])
```

---

## Tips & Best Practices

* Use **item dictionaries** with explicit labels — simpler and safer than managing parallel arrays.
* For early-stage users, prefer **wider priors** (e.g., `alpha=1, beta=1`) and a **hybrid** engine for balanced exploration.
* If you want faster convergence when the pool is calibrated, try **Bayes `max_info`** or **`ucb`**.
* Apply a **reasonable exposure cap** in large pools (e.g., `max_exposures_per_item=2`) to avoid overusing the same item.
* Inspect the returned Plotly HTML to sanity-check the posterior evolution during development.

---

## Numerical Notes

* Logits and probabilities are clipped to avoid under/overflow.
* Posterior normalization uses the trapezoidal rule; set `grid_points ≥ 101` (default 1001).
* The Beta prior is parameterized as `Beta(alpha, beta)` on `[0,1]`.

---

## Troubleshooting

* **ValueError: item keys missing** — ensure your items match the likelihood schema.
* **All candidates masked** — if using `max_exposures_per_item`, the runner will temporarily unmask all if everything is at cap.
* **Posterior looks flat** — try increasing `grid_points`, narrowing the prior, or presenting more discriminating items.
* **No Plotly output** — install `plotly` and ensure your environment can render/save HTML snippets.

---

## FAQ

**Q: Does `recommend()` update the posterior?**
A: No. It is side-effect free. Call `update_estimator(label, outcome)` after the user responds.

**Q: Can I plug in my own estimator?**
A: Yes. You can pass a pre-built `Bayesian2PL`-compatible estimator to the runner (advanced use). The `ItemResponsePrediction` wrapper focuses on the common case with its own internal runner.

**Q: How do I run pure bandit exploration?**
A: Initialize with `engine="bandit"` (the `policy` is ignored in that case) and tune `bandit_discount`, `bandit_c`, or pick a Beta-based strategy via the runner if you need.

---

## License

MIT License

Copyright (c) 2025 Mugalan
