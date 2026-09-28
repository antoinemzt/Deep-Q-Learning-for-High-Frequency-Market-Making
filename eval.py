"""
eval.py
=======
Evaluation module for the PPO market-making agent.

Provides:
  - AvellanedaStoikovPolicy  : closed-form optimal policy (A&S 2008)
  - RandomPolicy             : uniform random baseline
  - evaluate_policy          : runs N episodes, returns EvalResult
  - compare_policies         : side-by-side comparison table
  - print_report             : formatted console report

Usage (standalone)
------------------
    # Evaluate all three policies on the same LOB
    python eval.py

Usage (after training)
----------------------
    from eval import evaluate_policy, AvellanedaStoikovPolicy, compare_policies
    from stable_baselines3 import PPO

    model = PPO.load("runs/.../best_model/best_model")
    results = compare_policies(lob_df, env_cfg, model)
    print_report(results)

References
----------
    Avellaneda & Stoikov (2008). "High-frequency trading in a limit order book."
    Quantitative Finance, 8(3), 217-224.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from env.data_lob import LOBConfig, simulate
from env.MM_env import MarketMakingEnv, EnvConfig


# ---------------------------------------------------------------------------
# Policy protocol  (duck-typed — works with SB3 models and custom policies)
# ---------------------------------------------------------------------------

class Policy(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        ...


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov analytical policy
# ---------------------------------------------------------------------------

class AvellanedaStoikovPolicy:
    """
    Closed-form optimal quoting policy from Avellaneda & Stoikov (2008).

    The model derives the optimal bid/ask offsets as:

        δ_bid*(t) = γσ²(T-t) + (1/γ) ln(1 + γ/κ) - q·γσ²(T-t)
        δ_ask*(t) = γσ²(T-t) + (1/γ) ln(1 + γ/κ) + q·γσ²(T-t)

    Simplified (reservation price formulation):
        r(t) = s - q·γσ²(T-t)          ← reservation (indifference) price
        δ*   = γσ²(T-t)/2 + (1/γ)·ln(1 + γ/κ)   ← half-spread

        agent_bid = r - δ*
        agent_ask = r + δ*

    The action is then mapped to the nearest discrete grid level.

    Parameters
    ----------
    gamma       : risk-aversion coefficient (inventory penalty)
    sigma       : mid-price volatility (per step)
    kappa       : order arrival intensity decay (fill model)
    T           : episode horizon (steps)
    tick_size   : minimum price increment
    n_levels    : number of discrete offset levels (must match env)
    min_offset  : minimum offset in ticks
    max_offset  : maximum offset in ticks
    """

    def __init__(
        self,
        gamma:      float = 0.005,
        sigma:      float = 0.124,
        kappa:      float = 1.5,
        T:          int   = 2_000,
        tick_size:  float = 0.1,
        n_levels:   int   = 5,
        min_offset: int   = 1,
        max_offset: int   = 5,
    ) -> None:
        self.gamma      = gamma
        self.sigma      = sigma
        self.kappa      = kappa
        self.T          = T
        self.tick_size  = tick_size
        self.n_levels   = n_levels
        self.min_offset = min_offset
        self.max_offset = max_offset

    def _optimal_offsets(self, q: float, t: int) -> tuple[float, float]:
        """
        Compute the optimal bid/ask offsets in price units.

        Returns (delta_bid, delta_ask) as continuous values.
        """
        tau   = max(self.T - t, 1)          # time remaining
        gamma = self.gamma
        sigma = self.sigma
        kappa = self.kappa

        # Half-spread (symmetric component)
        half_spread = (gamma * sigma**2 * tau / 2.0
                       + (1.0 / gamma) * np.log(1.0 + gamma / kappa))

        # Inventory skew: positive inventory → push ask closer, bid further
        skew = q * gamma * sigma**2 * tau

        delta_bid = half_spread + skew    # agent wants to sell → widen bid
        delta_ask = half_spread - skew    # agent wants to sell → tighten ask

        # Enforce minimum spread of 1 tick on each side
        delta_bid = max(delta_bid, self.tick_size)
        delta_ask = max(delta_ask, self.tick_size)

        return delta_bid, delta_ask

    def _to_ticks(self, delta_price: float) -> int:
        """Convert a price offset to the nearest allowed tick level."""
        ticks = round(delta_price / self.tick_size)
        return int(np.clip(ticks, self.min_offset, self.max_offset))

    def _encode_action(self, bid_level: int, ask_level: int) -> int:
        """Encode (bid_level, ask_level) → flat action index."""
        bid_idx = bid_level - self.min_offset
        ask_idx = ask_level - self.min_offset
        return bid_idx * self.n_levels + ask_idx

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """
        obs layout (from MarketMakingEnv._get_obs):
            0  spread_norm
            1  imbalance
            2  vol_norm
            3  inventory_norm   ← q / max_inventory
            4  time_remaining   ← steps_left / episode_steps
            5  bid_offset_norm
            6  ask_offset_norm
        """
        inv_norm  = float(obs[3])
        time_rem  = float(obs[4])

        # Recover approximate raw values
        q = inv_norm * 100.0              # max_inventory = 100 (EnvConfig default)
        t = int((1.0 - time_rem) * self.T)

        delta_bid, delta_ask = self._optimal_offsets(q, t)
        bid_level = self._to_ticks(delta_bid)
        ask_level = self._to_ticks(delta_ask)

        action = self._encode_action(bid_level, ask_level)
        return int(action), None


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Uniform random action — establishes the performance floor."""

    def __init__(self, n_actions: int = 25, seed: int = 0) -> None:
        self.n_actions = n_actions
        self._rng      = np.random.default_rng(seed)

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, None]:
        return int(self._rng.integers(self.n_actions)), None


# ---------------------------------------------------------------------------
# Evaluation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    policy_name:    str
    n_episodes:     int

    # PnL
    pnl_mean:       float = 0.0
    pnl_std:        float = 0.0
    pnl_min:        float = 0.0
    pnl_max:        float = 0.0

    # Risk-adjusted
    sharpe:         float = 0.0         # mean / std of per-episode PnL
    sortino:        float = 0.0         # mean / std of negative PnL only

    # Execution
    fill_rate:      float = 0.0         # fills / steps
    trades_mean:    float = 0.0

    # Inventory
    inv_mean_abs:   float = 0.0         # mean |inventory| during episodes
    inv_final_abs:  float = 0.0         # |inventory| at episode end

    # Adverse selection proxy
    # Fraction of fills where mid moved against us next step
    adverse_sel_ratio: float = 0.0

    # Per-action distribution
    action_counts:  np.ndarray = field(default_factory=lambda: np.zeros(25))

    # Per-volatility-regime PnL (low / mid / high vol terciles)
    pnl_low_vol:    float = 0.0
    pnl_mid_vol:    float = 0.0
    pnl_high_vol:   float = 0.0

    elapsed_s:      float = 0.0


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy,
    lob_df:         pd.DataFrame,
    env_cfg:        EnvConfig,
    n_episodes:     int   = 20,
    policy_name:    str   = "policy",
    seed:           int   = 0,
    deterministic:  bool  = True,
) -> EvalResult:

    t0 = time.perf_counter()

    ep_pnls         : list[float] = []
    ep_rewards      : list[float] = []
    ep_trades       : list[int]   = []
    ep_inv_abs      : list[float] = []
    ep_inv_final    : list[float] = []
    ep_steps        : list[int]   = []
    ep_vol_tercile  : list[int]   = []
    action_counts   = np.zeros(env_cfg.n_quote_levels ** 2, dtype=np.int64)
    adverse_hits    = 0
    adverse_total   = 0

    # Pre-compute vol tercile boundaries
    spread_col = lob_df["ask_price_1"] - lob_df["bid_price_1"]
    vol_q33    = spread_col.quantile(0.33)
    vol_q66    = spread_col.quantile(0.66)

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        env = MarketMakingEnv(lob_df, env_cfg)
        obs, info = env.reset(seed=int(rng.integers(1_000_000)))

        ep_reward  = 0.0
        ep_pnl_raw = 0.0
        ep_inv_sum = 0.0
        prev_mid   = float(lob_df.loc[env._t, "mid"])
        last_bid_filled = False
        last_ask_filled = False
        step_count = 0

        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            action_counts[int(action)] += 1
            ep_reward  += reward
            ep_pnl_raw += info["step_pnl"]   # PnL brut sans pénalités
            ep_inv_sum += abs(info["inventory"])
            step_count += 1

            # Adverse selection
            cur_mid = float(lob_df.loc[env._t, "mid"])
            if last_bid_filled:
                adverse_total += 1
                if cur_mid < prev_mid:
                    adverse_hits += 1
            if last_ask_filled:
                adverse_total += 1
                if cur_mid > prev_mid:
                    adverse_hits += 1

            last_bid_filled = info["bid_filled"]
            last_ask_filled = info["ask_filled"]
            prev_mid = cur_mid

        # Vol tercile basé sur le spread moyen de l'épisode
        start_idx   = env._t - step_count
        end_idx     = env._t
        mean_spread = spread_col.iloc[start_idx:end_idx].mean()
        if mean_spread <= vol_q33:
            ep_vol_tercile.append(0)
        elif mean_spread <= vol_q66:
            ep_vol_tercile.append(1)
        else:
            ep_vol_tercile.append(2)

        ep_pnls.append(ep_pnl_raw)
        ep_rewards.append(ep_reward)
        ep_trades.append(info["trades"])
        ep_inv_abs.append(ep_inv_sum / max(step_count, 1))
        ep_inv_final.append(abs(info["inventory"]))
        ep_steps.append(step_count)

    ep_pnls_arr    = np.array(ep_pnls)
    ep_rewards_arr = np.array(ep_rewards)
    vols           = np.array(ep_vol_tercile)
    downside       = ep_rewards_arr[ep_rewards_arr < 0]

    def pnl_for_vol(v):
        mask = vols == v
        return float(ep_pnls_arr[mask].mean()) if mask.any() else 0.0

    return EvalResult(
        policy_name    = policy_name,
        n_episodes     = n_episodes,
        pnl_mean       = float(ep_pnls_arr.mean()),        # PnL brut
        pnl_std        = float(ep_pnls_arr.std()),
        pnl_min        = float(ep_pnls_arr.min()),
        pnl_max        = float(ep_pnls_arr.max()),
        sharpe         = float(ep_pnls_arr.mean() / (ep_rewards_arr.std() + 1e-9)),
        sortino        = float(ep_pnls_arr.mean() / (downside.std() + 1e-9))
                         if len(downside) > 1 else 0.0,
        fill_rate      = float(np.mean(ep_trades) / max(np.mean(ep_steps), 1)),
        trades_mean    = float(np.mean(ep_trades)),
        inv_mean_abs   = float(np.mean(ep_inv_abs)),
        inv_final_abs  = float(np.mean(ep_inv_final)),
        adverse_sel_ratio = float(adverse_hits / max(adverse_total, 1)),
        action_counts  = action_counts,
        pnl_low_vol    = pnl_for_vol(0),
        pnl_mid_vol    = pnl_for_vol(1),
        pnl_high_vol   = pnl_for_vol(2),
        elapsed_s      = time.perf_counter() - t0,
    )

# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def compare_policies(
    lob_df:      pd.DataFrame,
    env_cfg:     EnvConfig,
    ppo_model    = None,
    n_episodes:  int = 20,
    seed:        int = 0,
) -> dict[str, EvalResult]:
    """
    Evaluate three policies and return a dict of EvalResult.

    Parameters
    ----------
    ppo_model : SB3 PPO model (or None to skip)
    """
    results: dict[str, EvalResult] = {}

    # 1. Random baseline
    print("  Evaluating random baseline…")
    results["random"] = evaluate_policy(
        RandomPolicy(n_actions=env_cfg.n_quote_levels ** 2, seed=seed),
        lob_df, env_cfg,
        n_episodes  = n_episodes,
        policy_name = "Random",
        seed        = seed,
        deterministic = False,
    )

    # 2. A&S analytical policy
    print("  Evaluating Avellaneda-Stoikov policy…")
    results["as"] = evaluate_policy(
        AvellanedaStoikovPolicy(
            gamma      = env_cfg.inventory_penalty,
            sigma      = 0.124,              # measured from LOB (Phase 1)
            kappa      = 1.5,
            T          = env_cfg.episode_steps,
            tick_size  = 0.1,
            n_levels   = env_cfg.n_quote_levels,
            min_offset = env_cfg.min_offset_ticks,
            max_offset = env_cfg.max_offset_ticks,
        ),
        lob_df, env_cfg,
        n_episodes  = n_episodes,
        policy_name = "Avellaneda-Stoikov",
        seed        = seed,
    )

    # 3. PPO agent (if provided)
    if ppo_model is not None:
        print("  Evaluating PPO agent…")

        class SB3Wrapper:
            """Thin wrapper to unify SB3 predict API with our Policy protocol."""
            def __init__(self, model): self._m = model
            def predict(self, obs, deterministic=True):
                a, _ = self._m.predict(obs, deterministic=deterministic)
                return int(a), None

        results["ppo"] = evaluate_policy(
            SB3Wrapper(ppo_model),
            lob_df, env_cfg,
            n_episodes  = n_episodes,
            policy_name = "PPO",
            seed        = seed,
        )

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: dict[str, EvalResult]) -> None:
    policies = list(results.values())
    names    = [r.policy_name for r in policies]
    w        = max(len(n) for n in names) + 2

    sep = "─" * (w + 62)

    def row(label, *vals, fmt="{:>12.4f}"):
        cells = "".join(fmt.format(v) for v in vals)
        print(f"  {label:<28}{cells}")

    print()
    print("=" * (w + 64))
    print("  EVALUATION REPORT")
    print("=" * (w + 64))
    print(f"  {'':28}" + "".join(f"{n:>12}" for n in names))
    print(sep)

    print("  PnL")
    row("  mean (per episode)",   *[r.pnl_mean    for r in policies])
    row("  std",                  *[r.pnl_std     for r in policies])
    row("  min",                  *[r.pnl_min     for r in policies])
    row("  max",                  *[r.pnl_max     for r in policies])
    print(sep)

    print("  Risk-adjusted")
    row("  Sharpe",               *[r.sharpe      for r in policies])
    row("  Sortino",              *[r.sortino     for r in policies])
    print(sep)

    print("  Execution")
    row("  fill rate (%)",        *[r.fill_rate * 100 for r in policies])
    row("  trades / episode",     *[r.trades_mean for r in policies])
    print(sep)

    print("  Inventory")
    row("  mean |inv| (intra)",   *[r.inv_mean_abs   for r in policies])
    row("  mean |inv| (final)",   *[r.inv_final_abs  for r in policies])
    print(sep)

    print("  Adverse selection")
    row("  adv. sel. ratio (%)",  *[r.adverse_sel_ratio * 100 for r in policies])
    print(sep)

    print("  PnL by vol regime")
    row("  low  spread",          *[r.pnl_low_vol  for r in policies])
    row("  mid  spread",          *[r.pnl_mid_vol  for r in policies])
    row("  high spread",          *[r.pnl_high_vol for r in policies])
    print(sep)

    print("  Timing")
    row("  eval time (s)",        *[r.elapsed_s for r in policies], fmt="{:>12.2f}")
    print("=" * (w + 64))

    # Action distribution summary
    print()
    print("  Action distribution (% of steps)")
    n = int(np.sqrt(len(policies[0].action_counts)))
    print(f"  {'':20}" + "".join(f"  {p.policy_name:>16}" for p in policies))
    for r in policies:
        total = r.action_counts.sum()
        dist  = (r.action_counts / max(total, 1) * 100).reshape(n, n)
        dominant = np.unravel_index(r.action_counts.argmax(), (n, n))
        print(f"  {r.policy_name:<20}  dominant action: "
              f"bid-{dominant[0]+1}t / ask+{dominant[1]+1}t  "
              f"({dist[dominant]:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate market-making policies")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to saved PPO model (e.g. runs/.../best_model)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--lob-steps",type=int, default=200_000)
    parser.add_argument("--lob-seed", type=int, default=999,
                        help="Completely separate LOB seed for evaluation")
    args = parser.parse_args()

    print(f"\nSimulating evaluation LOB ({args.lob_steps:,} steps, seed={args.lob_seed})…")
    lob_df  = simulate(LOBConfig(n_steps=args.lob_steps, seed=args.lob_seed))
    env_cfg = EnvConfig(episode_steps=2_000, seed=args.seed)

    # Load PPO if provided
    ppo_model = None
    if args.model:
        try:
            from stable_baselines3 import PPO
            ppo_model = PPO.load(args.model)
            print(f"PPO model loaded from {args.model}")
        except Exception as e:
            print(f"[warn] Could not load PPO model: {e}")

    print(f"Running {args.episodes} episodes per policy…\n")
    results = compare_policies(lob_df, env_cfg,
                               ppo_model=ppo_model,
                               n_episodes=args.episodes,
                               seed=args.seed)
    print_report(results)
