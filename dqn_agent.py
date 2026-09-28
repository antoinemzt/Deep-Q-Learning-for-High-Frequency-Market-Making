import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim:        int,
        n_actions:      int,
        hidden:         list[int] = [256, 256],
        lr:             float     = 1e-3,
        gamma:          float     = 0.99,
        buffer_size:    int       = 100_000,
        batch_size:     int       = 256,
        target_update:  int       = 1_000,   # steps between target net sync
        eps_start:      float     = 1.0,
        eps_end:        float     = 0.05,
        eps_decay:      int       = 50_000,  # steps to decay epsilon
        device:         str       = 'cpu',
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.eps_start     = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.device        = torch.device(device)
        self.steps         = 0

        self.q_net      = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size, obs_dim)

    # ── Epsilon schedule ──────────────────────────────────────
    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps / self.eps_decay)

    # ── Action selection ──────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(obs_t).argmax(dim=1).item())

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Interface compatible avec eval.py."""
        if deterministic:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = int(self.q_net(obs_t).argmax(dim=1).item())
        else:
            action = self.select_action(obs)
        return action, None

    # ── Store transition ──────────────────────────────────────
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self.steps += 1

    # ── Learning step ─────────────────────────────────────────
    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        actions_t  = torch.LongTensor(actions).to(self.device)
        rewards_t  = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) courant
        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN : action choisie par q_net, évaluée par target_net
        with torch.no_grad():
            next_actions = self.q_net(next_obs_t).argmax(dim=1)
            next_q       = self.target_net(next_obs_t).gather(
                               1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Sync target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ── Save / Load ───────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            'q_net':      self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'steps':      self.steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps = ckpt['steps']