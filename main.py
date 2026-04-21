"""F1TENTH racing env + PPO on Warp, single-track dynamic vehicle model.

Force-based bicycle. State (7 floats):
    sx, sy, delta, u, psi, psip, v
    where u = body-frame longitudinal velocity, v = body-frame lateral velocity.

Tires: saturating tanh in lateral (linear at small slip, peaks at mu*F_z) plus
a friction-circle clip combining longitudinal demand with lateral force.
No kinematic fallback at low speed -- slip angles use atan2 with a clamped
denominator so tires are engaged at every speed and there is no free-grip
regime for the policy to exploit.
"""

import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import warp as wp
from cv2 import (
    COLOR_GRAY2RGB,
    IMREAD_GRAYSCALE,
    cvtColor,
    fillPoly,
    imread,
    polylines,
)
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from torch.distributions import Normal
from typer import run
from yaml import safe_load

import wandb

# ============================================================================
# Vehicle parameters (F1TENTH)
# ============================================================================
MASS = 3.74
LF = 0.15875  # CG -> front axle
LR = 0.17145  # CG -> rear axle
I_Z = 0.04712
H_CG = 0.074  # CG height (for longitudinal load transfer)
MU = 1.0489  # peak friction coefficient

# Dimensionless cornering stiffness coefficients.
# Linear stiffness (N/rad) = MU * F_z * C_*  -- the saturating tanh below
# uses C_* directly because the (MU*F_z) factor cancels.
C_F = 4.718
C_R = 5.4562

T_SB = 0.6  # brake split to front (RWD: drive only goes to rear)

STEER_MIN, STEER_MAX = -0.4189, 0.4189
STEER_V_MAX = 3.2
A_MAX = 9.51
V_MIN, V_MAX = -5.0, 20.0
V_SWITCH = 7.319

ATAN_EPS = 0.05  # clamp on |u| in slip-angle atan2 denominator

# ============================================================================
# Sim / obs / reward
# ============================================================================
DT = 1.0 / 60.0
SUBSTEPS = 2  # ST is much less stiff than MB; 2 is plenty
DT_SUB = DT / float(SUBSTEPS)
DT_SUB_HALF = DT_SUB * 0.5
DT_SUB_SIX = DT_SUB / 6.0
G = 9.81

WIDTH, LENGTH = 0.31, 0.58
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0))

NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
LIDAR_RANGE = 20.0
NUM_LOOKAHEAD = 10

OBS_PROP = 4  # delta, u, v, psip
OBS_LIDAR_OFF = OBS_PROP
OBS_FRENET_OFF = OBS_PROP + NUM_LIDAR
OBS_LOOK_OFF = OBS_FRENET_OFF + 2
OBS_DIM = OBS_LOOK_OFF + 2 * NUM_LOOKAHEAD
ACT_DIM = 2

MAX_STEPS = 10_000

# Original waypoint-progress reward
PROGRESS_SCALE = 100.0
PROGRESS_V_COEF = 10.0
WALL_PENALTY_COEF = 0.1
WALL_PENALTY_RATE = 3.0
TERM_PENALTY = 100.0

OCC_THRESH = 230
SMOOTH_WINDOW = 51
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DONE_TERMINATED = 1
DONE_TRUNCATED = 2

# ============================================================================
# Warp types and dynamics
# ============================================================================
# State layout (0-indexed):
#   0:sx  1:sy  2:delta  3:u  4:psi  5:psip  6:v
vec7 = wp.types.vector(length=7, dtype=wp.float32)


@wp.func
def reset7(sx: float, sy: float, psi: float) -> vec7:
    return vec7(sx, sy, 0.0, 0.0, psi, 0.0, 0.0)


@wp.func
def clamp_away(x: float, eps: float) -> float:
    """Push |x| up to >= eps, preserving sign; eps at x=0."""
    if x >= 0.0:
        return wp.max(x, eps)
    return wp.min(x, -eps)


@wp.func
def tire_forces(alpha: float, fx_demand: float, fz: float, c_stiff: float) -> wp.vec2:
    """Saturating-tanh lateral force + friction-circle clip on (F_x, F_y).

    Lateral force at small alpha:  fy ~ -MU*F_z * c_stiff * alpha   (linear)
    At large alpha:                fy -> -+ MU*F_z                  (saturated)
    The pair (fx_demand, fy_pure) is then scaled to fit inside the friction
    circle of radius MU*F_z, mixing longitudinal grip (drive/brake) with
    lateral grip (cornering) as the tire saturates.
    """
    f_max = MU * fz
    fy_pure = -f_max * wp.tanh(c_stiff * alpha)
    mag = wp.sqrt(fx_demand * fx_demand + fy_pure * fy_pure)
    scale = wp.where(mag > f_max, f_max / wp.max(mag, 1e-6), 1.0)
    return wp.vec2(fx_demand * scale, fy_pure * scale)


@wp.func
def st_deriv(s: vec7, vdelta: float, accel: float) -> vec7:
    delta = s[2]
    u = s[3]
    psi = s[4]
    psip = s[5]
    v = s[6]

    cd = wp.cos(delta)
    sd = wp.sin(delta)
    cp = wp.cos(psi)
    sp = wp.sin(psi)

    # --- Slip angles (atan2 with clamped denom; no kinematic fallback) ---
    u_safe = clamp_away(u, ATAN_EPS)
    alpha_f = wp.atan2(v + LF * psip, u_safe) - delta
    alpha_r = wp.atan2(v - LR * psip, u_safe)

    # --- Vertical loads with longitudinal load transfer ---
    fz_static_f = MASS * G * LR / (LF + LR)
    fz_static_r = MASS * G * LF / (LF + LR)
    dfz = MASS * accel * H_CG / (LF + LR)
    fz_f = wp.max(fz_static_f - dfz, 0.0)
    fz_r = wp.max(fz_static_r + dfz, 0.0)

    # --- Longitudinal force demand: RWD drive, split-brake ---
    fx_f_dem = wp.where(accel >= 0.0, 0.0, MASS * accel * T_SB)
    fx_r_dem = wp.where(accel >= 0.0, MASS * accel, MASS * accel * (1.0 - T_SB))

    # --- Per-axle tire forces (saturating + friction circle) ---
    front = tire_forces(alpha_f, fx_f_dem, fz_f, C_F)
    rear = tire_forces(alpha_r, fx_r_dem, fz_r, C_R)
    fx_f = front[0]
    fy_f = front[1]
    fx_r = rear[0]
    fy_r = rear[1]

    # --- Net body-frame force (front rotated by delta) ---
    fx_b = fx_r + fx_f * cd - fy_f * sd
    fy_b = fy_r + fy_f * cd + fx_f * sd

    return vec7(
        u * cp - v * sp,  # 0 dsx
        u * sp + v * cp,  # 1 dsy
        vdelta,  # 2 ddelta
        fx_b / MASS + psip * v,  # 3 du
        psip,  # 4 dpsi
        (LF * (fy_f * cd + fx_f * sd) - LR * fy_r) / I_Z,  # 5 dpsip
        fy_b / MASS - psip * u,  # 6 dv
    )


# ============================================================================
# Step kernel: input constraints + RK4 + crash/reward/reset + lidar + obs
# ============================================================================
@wp.kernel
def step_kernel(
    actions: wp.array(dtype=wp.vec2),
    observation: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    state: wp.array(dtype=vec7),
    step_count: wp.array(dtype=wp.int32),
    waypoint_idx: wp.array(dtype=wp.int32),
    origin: wp.vec2,
    res: float,
    dt_map: wp.array2d(dtype=wp.float32),
    cl_lut: wp.array2d(dtype=wp.int32),
    centerline: wp.array(dtype=wp.vec3),
    n_cl: int,
    look_step: int,
    lidar_dirs: wp.array(dtype=wp.vec2),
    seed_base: int,
):
    i = wp.tid()
    s = state[i]
    steps = step_count[i]
    wp_i = waypoint_idx[i]

    # --- Input constraints ---
    vdelta = wp.clamp(actions[i][0], -1.0, 1.0) * STEER_V_MAX
    if (vdelta < 0.0 and s[2] <= STEER_MIN) or (vdelta > 0.0 and s[2] >= STEER_MAX):
        vdelta = 0.0
    accel = wp.clamp(actions[i][1], -1.0, 1.0) * A_MAX
    abs_u = wp.abs(s[3])
    pos_lim = wp.where(abs_u > V_SWITCH, A_MAX * V_SWITCH / wp.max(abs_u, 0.1), A_MAX)
    accel = wp.clamp(accel, -A_MAX, pos_lim)
    if (s[3] <= V_MIN and accel < 0.0) or (s[3] >= V_MAX and accel > 0.0):
        accel = 0.0

    # --- RK4 substeps ---
    for _ in range(SUBSTEPS):
        k1 = st_deriv(s, vdelta, accel)
        k2 = st_deriv(s + k1 * DT_SUB_HALF, vdelta, accel)
        k3 = st_deriv(s + k2 * DT_SUB_HALF, vdelta, accel)
        k4 = st_deriv(s + k3 * DT_SUB, vdelta, accel)
        s = s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * DT_SUB_SIX
        s = vec7(
            s[0],
            s[1],
            wp.clamp(s[2], STEER_MIN, STEER_MAX),
            wp.clamp(s[3], V_MIN, V_MAX),
            s[4],
            s[5],
            s[6],
        )

    # --- Crash check / reward / waypoint ---
    origin_x = origin[0]
    origin_y = origin[1]
    mw = dt_map.shape[0]
    mh = dt_map.shape[1]
    mh_f = wp.float32(mh) - 1.0
    px = wp.clamp(wp.int32((s[0] - origin_x) / res), 0, mw - 1)
    py = wp.clamp(wp.int32(mh_f - (s[1] - origin_y) / res), 0, mh - 1)
    edt_val = dt_map[px, py] * res
    term = edt_val < CAR_HALF_DIAG
    trunc = steps >= MAX_STEPS
    steps += 1

    new_wp = cl_lut[px, py]
    d_wp = new_wp - wp_i
    if 2 * d_wp > n_cl:
        d_wp -= n_cl
    elif 2 * d_wp < -n_cl:
        d_wp += n_cl

    cth = centerline[new_wp][2]
    ch = wp.cos(s[4])
    sh = wp.sin(s[4])
    vx_w = s[3] * ch - s[6] * sh
    vy_w = s[3] * sh + s[6] * ch
    v_along = vx_w * wp.cos(cth) + vy_w * wp.sin(cth)

    progress = (
        wp.float32(d_wp)
        / wp.float32(n_cl)
        * PROGRESS_SCALE
        * (1.0 + wp.max(v_along, 0.0) / PROGRESS_V_COEF)
    )
    wall = -WALL_PENALTY_COEF * wp.exp(-WALL_PENALTY_RATE * edt_val)
    term_pen = wp.where(term, -TERM_PENALTY, 0.0)
    reward[i] = progress + wall + term_pen

    if term:
        done[i] = DONE_TERMINATED
    elif trunc:
        done[i] = DONE_TRUNCATED
    else:
        done[i] = 0

    # --- Reset on term/trunc ---
    if term or trunc:
        rng = wp.rand_init(seed_base + i * 73 + steps * 31)
        rnd = wp.int32(wp.randf(rng) * wp.float32(n_cl)) % n_cl
        rpt = centerline[rnd]
        s = reset7(rpt[0], rpt[1], rpt[2])
        steps = 0
        new_wp = rnd
        ch = wp.cos(s[4])
        sh = wp.sin(s[4])

    # --- LIDAR raycast on EDT (sphere-tracing) ---
    lx = s[0] + LF * ch
    ly = s[1] + LF * sh
    lpx = wp.clamp(wp.int32((lx - origin_x) / res), 0, mw - 1)
    lpy = wp.clamp(wp.int32(mh_f - (ly - origin_y) / res), 0, mh - 1)
    lpos = wp.vec2(wp.float32(lpx), wp.float32(lpy))
    lrange_px = LIDAR_RANGE / res
    for j in range(lidar_dirs.shape[0]):
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        dpx = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        ray = lpos
        dist = float(0.0)
        while dist < lrange_px:
            rx = wp.int32(ray[0])
            ry = wp.int32(ray[1])
            if rx < 0 or rx >= mw or ry < 0 or ry >= mh:
                break
            step_px = dt_map[rx, ry]
            ray = ray + dpx * step_px
            dist += step_px
            if step_px == 0.0:
                break
        observation[i, OBS_LIDAR_OFF + j] = wp.min(dist, lrange_px) * res

    # --- Frenet + lookahead ---
    cpt = centerline[new_wp]
    cx_p = cpt[0]
    cy_p = cpt[1]
    cth_p = cpt[2]
    scth = wp.sin(cth_p)
    cct = wp.cos(cth_p)
    heading_err = wp.atan2(scth * ch - cct * sh, cct * ch + scth * sh)
    lateral_err = -(s[0] - cx_p) * scth + (s[1] - cy_p) * cct
    observation[i, OBS_FRENET_OFF] = heading_err
    observation[i, OBS_FRENET_OFF + 1] = lateral_err

    idx = new_wp
    for k in range(NUM_LOOKAHEAD):
        idx += look_step
        if idx >= n_cl:
            idx -= n_cl
        wpt = centerline[idx]
        dx = wpt[0] - s[0]
        dy = wpt[1] - s[1]
        observation[i, OBS_LOOK_OFF + k * 2] = dx * ch + dy * sh
        observation[i, OBS_LOOK_OFF + k * 2 + 1] = -dx * sh + dy * ch

    # --- Proprioceptive: delta, u, v, psip ---
    observation[i, 0] = s[2]
    observation[i, 1] = s[3]
    observation[i, 2] = s[6]
    observation[i, 3] = s[5]

    state[i] = s
    step_count[i] = steps
    waypoint_idx[i] = new_wp


# ============================================================================
# Map + centerline
# ============================================================================
class Map:
    def __init__(self, map_path: str):
        p = Path(map_path)
        cfg = safe_load(p.with_suffix(".yaml").read_text())
        self.origin = tuple(cfg["origin"][:2])
        self.resolution = float(cfg["resolution"])
        img = imread(str(p.parent / cfg["image"]), IMREAD_GRAYSCALE)
        self.img = img
        occ = img < OCC_THRESH  # True = wall
        self.edt = distance_transform_edt(~occ).astype(np.float32).T  # (w, h)
        skel = skeletonize(~occ)
        self.centerline = self._trace(skel)
        self.kdtree = KDTree(self.centerline[:, :2])
        self.lut = self._build_lut(occ.shape)

    def _trace(self, skel):
        ys, xs = np.where(skel)
        if len(xs) == 0:
            raise RuntimeError("empty skeleton")
        coords = set(zip(xs.tolist(), ys.tolist()))
        start = (int(xs[0]), int(ys[0]))
        path = [start]
        cur = start
        while True:
            coords.discard(cur)
            nxt = None
            for dx, dy in ADJ:
                c = (cur[0] + dx, cur[1] + dy)
                if c in coords:
                    nxt = c
                    break
            if nxt is None:
                break
            cur = nxt
            path.append(cur)
        arr = np.array(path, dtype=np.float32)
        h = self.img.shape[0]
        wx = self.origin[0] + arr[:, 0] * self.resolution
        wy = self.origin[1] + (h - 1 - arr[:, 1]) * self.resolution
        if len(wx) > SMOOTH_WINDOW:
            wx = savgol_filter(wx, SMOOTH_WINDOW, 3, mode="wrap")
            wy = savgol_filter(wy, SMOOTH_WINDOW, 3, mode="wrap")
        dx = np.roll(wx, -1) - np.roll(wx, 1)
        dy = np.roll(wy, -1) - np.roll(wy, 1)
        theta = np.arctan2(dy, dx)
        return np.stack([wx, wy, theta], axis=1).astype(np.float32)

    def _build_lut(self, occ_shape):
        h, w = occ_shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        wx = self.origin[0] + xs * self.resolution
        wy = self.origin[1] + (h - 1 - ys) * self.resolution
        pts = np.stack([wx.ravel(), wy.ravel()], axis=1)
        _, idx = self.kdtree.query(pts, k=1)
        return idx.reshape(h, w).astype(np.int32).T


# ============================================================================
# Vectorised racing env (does not inherit from gym.vector.VectorEnv;
# Gymnasium 1.x changed that __init__ signature and we don't use any of
# its methods anyway)
# ============================================================================
class RacingEnv:
    def __init__(self, map_path: str, num_envs: int, device: str, seed: int = 0):
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.device = device
        self.map = Map(map_path)
        d = device
        cl = self.map.centerline
        n_cl = len(cl)

        self.state = wp.zeros(num_envs, dtype=vec7, device=d)
        self.step_count = wp.full(num_envs, MAX_STEPS, dtype=wp.int32, device=d)
        self.waypoint = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.obs_wp = wp.zeros((num_envs, OBS_DIM), dtype=wp.float32, device=d)
        self.rew_wp = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.done_wp = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.act_wp = wp.zeros(num_envs, dtype=wp.vec2, device=d)

        self.dt_map = wp.from_numpy(self.map.edt, dtype=wp.float32, device=d)
        self.cl_lut = wp.from_numpy(self.map.lut, dtype=wp.int32, device=d)
        self.centerline = wp.from_numpy(cl, dtype=wp.vec3, device=d)
        self.n_cl = n_cl

        fov = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        dirs = np.stack([np.cos(fov), np.sin(fov)], axis=1).astype(np.float32)
        self.lidar_dirs = wp.from_numpy(dirs, dtype=wp.vec2, device=d)

        rough_len = float(np.sum(np.linalg.norm(np.diff(cl[:, :2], axis=0), axis=1)))
        self.look_step = max(1, int(0.6 * n_cl / max(rough_len, 1.0)))

        self.obs_buf = wp.to_torch(self.obs_wp)
        self.rew_buf = wp.to_torch(self.rew_wp)
        self.done_buf = wp.to_torch(self.done_wp)
        self.act_buf = wp.to_torch(self.act_wp)
        self.state_buf = wp.to_torch(self.state).view(num_envs, 7)
        self.step_buf = wp.to_torch(self.step_count)
        self.wp_buf = wp.to_torch(self.waypoint)
        self._zero_act = torch.zeros(num_envs, ACT_DIM, device=d, dtype=torch.float32)

        self._seed = seed

    def reset(self, *_, **__):
        self.step_buf.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        return self.obs_buf.clone(), {}

    def step(self, action):
        self._launch(action)
        return (
            self.obs_buf.clone(),
            self.rew_buf.clone(),
            self.done_buf == DONE_TERMINATED,
            self.done_buf == DONE_TRUNCATED,
            {},
        )

    def _launch(self, action):
        self._sanitize()
        self.act_buf.copy_(action.to(self.device, dtype=torch.float32))
        self._seed = (self._seed + 1) & 0x7FFFFFFF
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                self.act_wp,
                self.obs_wp,
                self.rew_wp,
                self.done_wp,
                self.state,
                self.step_count,
                self.waypoint,
                wp.vec2(*self.map.origin),
                float(self.map.resolution),
                self.dt_map,
                self.cl_lut,
                self.centerline,
                self.n_cl,
                self.look_step,
                self.lidar_dirs,
                self._seed,
            ],
            device=self.device,
        )

    def _sanitize(self):
        bad = ~torch.isfinite(self.state_buf).all(dim=1)
        if bad.any():
            self.step_buf[bad] = MAX_STEPS
            self.state_buf[bad] = 0.0

    def save_state(self):
        return (
            self.state_buf.clone(),
            self.step_buf.clone(),
            self.wp_buf.clone(),
            self.obs_buf.clone(),
            self.rew_buf.clone(),
            self.done_buf.clone(),
        )

    def restore_state(self, snap):
        s, sc, wpi, obs, rew, dn = snap
        self.state_buf.copy_(s)
        self.step_buf.copy_(sc)
        self.wp_buf.copy_(wpi)
        self.obs_buf.copy_(obs)
        self.rew_buf.copy_(rew)
        self.done_buf.copy_(dn)


# ============================================================================
# PPO building blocks
# ============================================================================
class RunningMeanStd:
    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps
        self.inv_std = torch.ones(shape, device=device)

    def update(self, x):
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        n = x.shape[0]
        delta = m - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * (n / tot)
        m_a = self.var * self.count
        m_b = v * n
        self.var = (m_a + m_b + delta.pow(2) * self.count * n / tot) / tot
        self.count = tot
        self.inv_std = 1.0 / (self.var.sqrt() + 1e-6)


class ReturnNormalizer:
    def __init__(self, num_envs, gamma, device):
        self.ret = torch.zeros(num_envs, device=device)
        self.rms = RunningMeanStd((), device)
        self.gamma = gamma

    def __call__(self, reward, done):
        self.ret = self.ret * self.gamma * (~done).float() + reward
        self.rms.update(self.ret)
        return reward * self.rms.inv_std


def _init_layer(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.actor_net = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden, act_dim), gain=0.01),
        )
        self.critic_net = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden, 1), gain=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def dist(self, x):
        return Normal(self.actor_net(x), self.log_std.exp())

    def value(self, x):
        return self.critic_net(x).squeeze(-1)

    def act(self, x):
        d = self.dist(x)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(x)


class KLAdaptiveLR:
    def __init__(self, optimizer, target_kl=0.01, factor=1.5, lr_min=1e-6, lr_max=1e-2):
        self.opt = optimizer
        self.target = target_kl
        self.factor = factor
        self.lr_min = lr_min
        self.lr_max = lr_max

    def step(self, kl):
        lr = self.opt.param_groups[0]["lr"]
        if kl > self.target * 2:
            lr /= self.factor
        elif kl < self.target / 2:
            lr *= self.factor
        lr = float(np.clip(lr, self.lr_min, self.lr_max))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


# ============================================================================
# Rollout video
# ============================================================================
def record_rollout(env, agent, obs_rms, out_path, num_steps=900, env_idx=0):
    snap = env.save_state()
    obs, _ = env.reset()
    frames = []
    with torch.no_grad():
        for _ in range(num_steps):
            on = (obs - obs_rms.mean) * obs_rms.inv_std
            a = agent.dist(on).mean.clamp(-1, 1)
            obs, _, _, _, _ = env.step(a)
            sx = env.state_buf[env_idx, 0].item()
            sy = env.state_buf[env_idx, 1].item()
            psi = env.state_buf[env_idx, 4].item()
            frames.append(_draw_frame(env.map, sx, sy, psi))
    env.restore_state(snap)
    if frames:
        imageio.mimsave(out_path, frames, fps=30)


def _draw_frame(m, sx, sy, psi):
    img = cvtColor(m.img, COLOR_GRAY2RGB)
    h = img.shape[0]
    pts = m.centerline[:, :2]
    ui = ((pts[:, 0] - m.origin[0]) / m.resolution).astype(np.int32)
    vi = (h - 1 - (pts[:, 1] - m.origin[1]) / m.resolution).astype(np.int32)
    polylines(img, [np.stack([ui, vi], axis=1)], False, (0, 180, 0), 1)
    c, s = np.cos(psi), np.sin(psi)
    corners = np.array(
        [
            [LENGTH / 2, WIDTH / 2],
            [LENGTH / 2, -WIDTH / 2],
            [-LENGTH / 2, -WIDTH / 2],
            [-LENGTH / 2, WIDTH / 2],
        ],
        dtype=np.float32,
    )
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    world = corners @ R.T + np.array([sx, sy])
    px = ((world[:, 0] - m.origin[0]) / m.resolution).astype(np.int32)
    py = (h - 1 - (world[:, 1] - m.origin[1]) / m.resolution).astype(np.int32)
    fillPoly(img, [np.stack([px, py], axis=1)], (40, 40, 220))
    return img


# ============================================================================
# PPO training
# ============================================================================
def train(
    map_path: str = "maps/berlin",
    num_envs: int = 4096,
    total_steps: int = 20_000_000,
    rollout_len: int = 64,
    epochs: int = 4,
    minibatches: int = 8,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    video_every: int = 50,
    project: str = "f1tenth-st",
):
    wp.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project=project, config=locals())
    env = RacingEnv(map_path, num_envs, device)

    agent = Agent(OBS_DIM, ACT_DIM).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    lr_sched = KLAdaptiveLR(opt)

    obs_rms = RunningMeanStd((OBS_DIM,), device)
    ret_norm = ReturnNormalizer(num_envs, gamma, device)

    obs_buf = torch.zeros(rollout_len, num_envs, OBS_DIM, device=device)
    act_buf = torch.zeros(rollout_len, num_envs, ACT_DIM, device=device)
    logp_buf = torch.zeros(rollout_len, num_envs, device=device)
    rew_buf = torch.zeros(rollout_len, num_envs, device=device)
    done_buf = torch.zeros(rollout_len, num_envs, device=device)
    val_buf = torch.zeros(rollout_len, num_envs, device=device)

    obs, _ = env.reset()
    ep_returns = deque(maxlen=100)
    cur_return = torch.zeros(num_envs, device=device)
    global_step = 0
    iteration = 0
    start = time.time()

    pi_loss = v_loss = ent = torch.zeros((), device=device)

    while global_step < total_steps:
        iteration += 1
        for t in range(rollout_len):
            obs_rms.update(obs)
            on = (obs - obs_rms.mean) * obs_rms.inv_std
            with torch.no_grad():
                a, logp, val = agent.act(on)
            obs_buf[t] = on
            act_buf[t] = a
            logp_buf[t] = logp
            val_buf[t] = val
            obs, rew, term, trunc, _ = env.step(a.clamp(-1, 1))
            done = term | trunc
            rew_n = ret_norm(rew, done)
            rew_buf[t] = rew_n
            done_buf[t] = done.float()
            cur_return += rew
            done_idx = torch.where(done)[0].tolist()
            for i in done_idx:
                ep_returns.append(cur_return[i].item())
                cur_return[i] = 0.0
            global_step += num_envs

        with torch.no_grad():
            on = (obs - obs_rms.mean) * obs_rms.inv_std
            next_val = agent.value(on)

        adv = torch.zeros_like(rew_buf)
        last = torch.zeros(num_envs, device=device)
        for t in reversed(range(rollout_len)):
            nv = next_val if t == rollout_len - 1 else val_buf[t + 1]
            nm = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * nv * nm - val_buf[t]
            last = delta + gamma * gae_lambda * nm * last
            adv[t] = last
        ret = adv + val_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bs = rollout_len * num_envs
        mb_size = bs // minibatches
        flat_obs = obs_buf.reshape(bs, OBS_DIM)
        flat_act = act_buf.reshape(bs, ACT_DIM)
        flat_logp = logp_buf.reshape(bs)
        flat_adv = adv.reshape(bs)
        flat_ret = ret.reshape(bs)

        kls = []
        for _ in range(epochs):
            idx = torch.randperm(bs, device=device)
            for m in range(minibatches):
                mb = idx[m * mb_size : (m + 1) * mb_size]
                d = agent.dist(flat_obs[mb])
                new_logp = d.log_prob(flat_act[mb]).sum(-1)
                ratio = (new_logp - flat_logp[mb]).exp()
                p1 = ratio * flat_adv[mb]
                p2 = ratio.clamp(1 - clip, 1 + clip) * flat_adv[mb]
                pi_loss = -torch.min(p1, p2).mean()
                v_loss = (agent.value(flat_obs[mb]) - flat_ret[mb]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(0.5 * (flat_logp[mb] - new_logp).pow(2).mean().item())

        mean_kl = float(np.mean(kls))
        new_lr = lr_sched.step(mean_kl)

        elapsed = time.time() - start
        wandb.log(
            {
                "step": global_step,
                "sps": global_step / max(elapsed, 1e-6),
                "pi_loss": pi_loss.item(),
                "v_loss": v_loss.item(),
                "entropy": ent.item(),
                "kl": mean_kl,
                "lr": new_lr,
                "ret_mean": float(np.mean(ep_returns) if ep_returns else 0.0),
                "ret_norm_std": float(ret_norm.rms.var.sqrt()),
            }
        )

        if iteration % video_every == 0:
            out = f"/tmp/rollout_{global_step}.mp4"
            record_rollout(env, agent, obs_rms, out)
            wandb.log(
                {"rollout": wandb.Video(out, fps=30, format="mp4")}, step=global_step
            )

    wandb.finish()
    return global_step


def main(
    map_path: str = "maps/berlin",
    num_envs: int = 4096,
    total_steps: int = 20_000_000,
):
    train(map_path=map_path, num_envs=num_envs, total_steps=total_steps)


if __name__ == "__main__":
    run(main)
