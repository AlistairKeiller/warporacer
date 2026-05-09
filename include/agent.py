import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
#import wandb
from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines
from torch.distributions import Normal

from include.constants import *


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape).float()
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta, alpha=bc / tot)
        self.var = (
            self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)
        ) / tot
        self.count = tot
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, x, clip: float = 10.0):
        return ((x - self.mean) * self.inv_std).clamp(-clip, clip)


class ReturnNormalizer:
    def __init__(self, num_envs, gamma, device):
        self.gamma = gamma
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms = RunningMeanStd((), device)

    def update(self, reward, done):
        self.returns = self.returns * self.gamma * (1.0 - done) + reward
        self.rms.update(self.returns)

    def normalize(self, reward):
        return reward * self.rms.inv_std


def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    LOGSTD_MIN, LOGSTD_MAX = -1.6, -0.3

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.full((1, act_dim), -0.5))

    def _dist(self, obs):
        mean = self.actor(obs)
        ls = self.log_std.expand_as(mean).clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return Normal(mean, ls.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act_value(self, obs, action=None):
        d = self._dist(obs)
        if action is None:
            action = d.sample()
        return action, d.log_prob(action).sum(-1), d.entropy().sum(-1), self.value(obs)

    def deterministic(self, obs):
        return self.actor(obs)


class KLAdaptiveLR:
    def __init__(self, opt, target_kl=0.02, factor=1.5, lr_min=1e-6, lr_max=3e-3):
        self.opt = opt
        self.target = target_kl
        self.factor = factor
        self.lr_min = lr_min
        self.lr_max = lr_max

    def step(self, kl):
        for pg in self.opt.param_groups:
            lr = pg["lr"]
            if kl > 2.0 * self.target:
                pg["lr"] = max(self.lr_min, lr / self.factor)
            elif kl < 0.5 * self.target:
                pg["lr"] = min(self.lr_max, lr * self.factor)

    @property
    def lr(self):
        return self.opt.param_groups[0]["lr"]


# Rollout video
# TODO: Implement this in Warp Render instead
def record_rollout(env, agent, num_steps, out_path, obs_rms=None):
    snap = env.save_state()
    was_training = agent.training
    agent.eval()
    try:
        m = env.map
        corners = np.array(
            [
                [-LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, WIDTH / 2],
                [-LENGTH / 2, WIDTH / 2],
            ]
        )

        def w2p(x, y):
            return int((x - m.ox) / m.res), int(m.h - 1 - (y - m.oy) / m.res)

        trail = deque(maxlen=300)
        raw, _ = env.reset()
        obs = obs_rms.normalize(raw) if obs_rms else raw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(out_path), fps=int(1 / DT), macro_block_size=1
        ) as w:
            with torch.no_grad():
                for _ in range(num_steps):
                    a = agent.deterministic(obs)
                    raw, _, term, trunc, _ = env.step(a)
                    obs = obs_rms.normalize(raw) if obs_rms else raw
                    row = env.cars_buf[0].tolist()
                    x, y, psi = row[0], row[1], row[4]
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        trail.clear()
                    trail.append((x, y))
                    frame = cvtColor(m.raw, COLOR_GRAY2RGB)
                    if len(trail) > 1:
                        polylines(
                            frame,
                            [np.array([w2p(*p) for p in trail], dtype=np.int32)],
                            False,
                            (0, 200, 0),
                            2,
                        )
                    R = np.array(
                        [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                    )
                    world = corners @ R.T + (x, y)
                    fillPoly(
                        frame,
                        [np.array([w2p(*p) for p in world], dtype=np.int32)],
                        (255, 50, 50),
                    )
                    w.append_data(frame)
    finally:
        env.restore_state(snap)
        agent.train(was_training)


# PPO training
def train(
    env,
    agent,
    iterations=2000,
    rollouts=24,
    epochs=5,
    minibatches=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    vf_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    max_grad_norm=0.5,
    lr=3e-4,
    target_kl=0.02,
    log_dir=Path("./logs"),
    record_every=100,
    record_steps=1800,
):
    device = next(agent.parameters()).device
    N = env.num_envs
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    sched = KLAdaptiveLR(opt, target_kl=target_kl)
    obs_rms = RunningMeanStd((OBS_DIM,), device)
    ret_rms = ReturnNormalizer(N, gamma, device)

    obs_b = torch.zeros((rollouts, N, OBS_DIM), device=device)
    act_b = torch.zeros((rollouts, N, ACT_DIM), device=device)
    logp_b = torch.zeros((rollouts, N), device=device)
    rew_b = torch.zeros((rollouts, N), device=device)
    done_b = torch.zeros((rollouts, N), device=device)
    term_b = torch.zeros((rollouts, N), device=device)
    val_b = torch.zeros((rollouts, N), device=device)

    raw, _ = env.reset()
    obs_rms.update(raw)
    obs = obs_rms.normalize(raw)
    ep_ret = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    finished_rets, finished_lens = deque(maxlen=100), deque(maxlen=100)

    global_step = 0
    t0 = time.time()
    last_t = t0

    for it in range(iterations):
        agent.eval()
        with torch.no_grad():
            for t in range(rollouts):
                obs_b[t] = obs
                act, logp, _, val = agent.act_value(obs)
                act_b[t] = act
                logp_b[t] = logp
                val_b[t] = val
                raw, raw_rew, term, trunc, _ = env.step(act)
                env.vs.render() # Live rendering of training
                done = (term | trunc).float()
                ret_rms.update(raw_rew, done)
                rew_b[t] = ret_rms.normalize(raw_rew)
                done_b[t] = done
                term_b[t] = term.float()
                ep_ret.add_(raw_rew)
                ep_len.add_(1.0)
                fin = done.bool()
                if fin.any():
                    finished_rets.extend(ep_ret[fin].cpu().tolist())
                    finished_lens.extend(ep_len[fin].cpu().tolist())
                    ep_ret[fin] = 0.0
                    ep_len[fin] = 0.0
                obs_rms.update(raw)
                obs = obs_rms.normalize(raw)
            next_val = agent.value(obs)

        # GAE
        val_ext = torch.cat([val_b, next_val.unsqueeze(0)], 0)
        adv_b = torch.zeros_like(rew_b)
        last = torch.zeros_like(next_val)
        for t in reversed(range(rollouts)):
            nonterm = 1.0 - term_b[t]
            nondone = 1.0 - done_b[t]
            delta = rew_b[t] + gamma * val_ext[t + 1] * nonterm - val_b[t]
            last = delta + gamma * gae_lambda * nondone * last
            adv_b[t] = last
        ret_b = adv_b + val_b
        global_step += rollouts * N

        # Flatten
        B = rollouts * N
        b_obs = obs_b.reshape(B, OBS_DIM)
        b_act = act_b.reshape(B, ACT_DIM)
        b_logp = logp_b.reshape(B)
        b_adv = adv_b.reshape(B)
        b_ret = ret_b.reshape(B)
        b_val = val_b.reshape(B)
        mb = B // minibatches

        agent.train()
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_upd = 0
        kl_stop = False
        for epoch in range(epochs):
            perm = torch.randperm(B, device=device)
            epoch_kl = 0.0
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                _, new_logp, ent, new_val = agent.act_value(b_obs[idx], b_act[idx])
                logratio = new_logp - b_logp[idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > clip).float().mean().item()
                epoch_kl += approx_kl
                adv_mb = b_adv[idx]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                s1 = ratio * adv_mb
                s2 = ratio.clamp(1 - clip, 1 + clip) * adv_mb
                pg = -torch.min(s1, s2).mean()

                v_err = new_val - b_ret[idx]
                if vf_clip > 0:
                    v_clipped = b_val[idx] + (new_val - b_val[idx]).clamp(
                        -vf_clip, vf_clip
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            v_err.square(), (v_clipped - b_ret[idx]).square()
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * v_err.square().mean()
                ent_m = ent.mean()
                loss = pg + vf_coef * v_loss - ent_coef * ent_m
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
                with torch.no_grad():
                    agent.log_std.clamp_(Agent.LOGSTD_MIN, Agent.LOGSTD_MAX)
                stats["pg"] += pg.item()
                stats["v"] += v_loss.item()
                stats["ent"] += ent_m.item()
                stats["kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                n_upd += 1
            if epoch_kl / max(minibatches, 1) > 1.5 * target_kl:
                kl_stop = True
                break
        for k in stats:
            stats[k] /= max(n_upd, 1)
        sched.step(stats["kl"])

        now = time.time()
        sps = int(rollouts * N / max(now - last_t, 1e-9))
        last_t = now
        log = {
            "policy_loss": stats["pg"],
            "value_loss": stats["v"],
            "entropy": stats["ent"],
            "approx_kl": stats["kl"],
            "clipfrac": stats["clipfrac"],
            "kl_stop": int(kl_stop),
            "log_std": agent.log_std.mean().item(),
            "lr": sched.lr,
            "sps": sps,
            "iteration": it,
        }
        if finished_rets:
            log["ep_return"] = float(np.mean(finished_rets))
            log["ep_length"] = float(np.mean(finished_lens))

        print(f"wandb.log: global_step={global_step}")
        # TODO: Implement W&B
        # try:
        #     wandb.log(log, step=global_step)
        # except Exception:
        #     pass

        if it % 10 == 0:
            er = log.get("ep_return", float("nan"))
            print(
                f"[it {it:4d}] step={global_step:>9d} sps={sps:>6d} "
                f"ret={er:8.2f} kl={stats['kl']:.4f} lr={sched.lr:.2e}"
                f"{' KL-STOP' if kl_stop else ''}"
            )
        if record_every > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            print(f"record_rollout: out={out}")
            # TODO: Implement W&B
            # try:
            #     record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
            #     wandb.log(
            #         {"rollout": wandb.Video(str(out), format="mp4")}, step=global_step
            #     )
            # except Exception as e:
            #     print(f"[rollout {it + 1}] failed: {e}")
    return time.time() - t0, obs_rms, ret_rms, global_step
