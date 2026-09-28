import numpy as np
import torch
from collections import deque
from pathlib import Path
from datetime import datetime

from env.data_lob import LOBConfig, simulate
from env.MM_env import MarketMakingEnv, EnvConfig
from dqn_agent import DQNAgent


def train_dqn(
    total_steps:    int   = 500_000,
    lob_seed:       int   = 42,
    eval_lob_seed:  int   = 99,
    episode_steps:  int   = 2_000,
    hidden:         list  = [256, 256],
    lr:             float = 1e-3,
    gamma:          float = 0.99,
    buffer_size:    int   = 100_000,
    batch_size:     int   = 256,
    target_update:  int   = 1_000,
    eps_start:      float = 1.0,
    eps_end:        float = 0.05,
    eps_decay:      int   = 50_000,
    learn_every:    int   = 4,        # gradient step tous les N steps
    eval_every:     int   = 20_000,
    eval_episodes:  int   = 10,
    save_dir:       str   = 'runs_dqn',
    device:         str   = 'cpu',
    seed:           int   = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Data ──────────────────────────────────────────────────
    print("Simulating LOB...")
    lob_train = simulate(LOBConfig(n_steps=500_000, seed=lob_seed))
    lob_eval  = simulate(LOBConfig(n_steps=100_000, seed=eval_lob_seed))

    env_cfg  = EnvConfig(episode_steps=episode_steps, seed=seed)
    env      = MarketMakingEnv(lob_train, env_cfg)
    obs_dim  = env.observation_space.shape[0]
    n_actions= env.action_space.n

    # ── Agent ─────────────────────────────────────────────────
    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=n_actions, hidden=hidden,
        lr=lr, gamma=gamma, buffer_size=buffer_size,
        batch_size=batch_size, target_update=target_update,
        eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
        device=device,
    )

    # ── Output dir ────────────────────────────────────────────
    run_id  = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(save_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best_reward = -np.inf

    # ── Logging ───────────────────────────────────────────────
    ep_rewards      = deque(maxlen=50)
    ep_losses       = deque(maxlen=200)
    action_counts   = np.zeros(n_actions, dtype=np.int64)

    log_steps        = []
    log_mean_rewards = []
    log_mean_losses  = []

    global_step = 0
    ep_reward   = 0.0
    ep_num      = 0

    obs, _ = env.reset(seed=seed)

    print(f"Training DQN — {total_steps:,} steps | obs_dim={obs_dim} | n_actions={n_actions}")
    print(f"Output: {out_dir}\n")

    # ── Main loop ─────────────────────────────────────────────
    while global_step < total_steps:
        action = agent.select_action(obs)
        action_counts[action] += 1
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        ep_reward   += reward
        global_step += 1
        obs          = next_obs

        if global_step % learn_every == 0:
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

        if done:
            ep_rewards.append(ep_reward)
            ep_num   += 1
            ep_reward = 0.0
            obs, _    = env.reset()

        if global_step % 10_000 == 0:
            mean_r = np.mean(ep_rewards) if ep_rewards else 0
            mean_l = np.mean(ep_losses)  if ep_losses  else 0
            log_steps.append(global_step)
            log_mean_rewards.append(mean_r)
            log_mean_losses.append(mean_l)
            print(f"step {global_step:>7,} | ep {ep_num:>4} | "
                  f"reward {mean_r:>10.1f} | loss {mean_l:.4f} | "
                  f"eps {agent.epsilon:.3f} | buffer {len(agent.buffer):,}")

        if global_step % eval_every == 0:
            eval_reward = evaluate(agent, lob_eval, env_cfg, eval_episodes)
            print(f"  → EVAL step {global_step:,} | mean_reward={eval_reward:.1f}")
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(str(out_dir / 'best_model.pt'))
                print(f"  → ✓ New best: {best_reward:.1f}")

    # ── Save final ────────────────────────────────────────────
    agent.save(str(out_dir / 'final_model.pt'))
    print(f"\nDone. Best eval reward: {best_reward:.1f}")
    print(f"Models saved in {out_dir}")

    logs = {
        'steps':         np.array(log_steps),
        'mean_rewards':  np.array(log_mean_rewards),
        'mean_losses':   np.array(log_mean_losses),
        'action_counts': action_counts,
    }
    return agent, str(out_dir), logs


def evaluate(agent, lob_df, env_cfg, n_episodes=10) -> float:
    rewards = []
    for ep in range(n_episodes):
        env = MarketMakingEnv(lob_df, env_cfg)
        obs, _ = env.reset(seed=ep)
        cum_r, done = 0.0, False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            cum_r += r
            done = term or trunc
        rewards.append(cum_r)
    return float(np.mean(rewards))


