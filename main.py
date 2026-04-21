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

OCC_THRESH = 230
SMOOTH_WINDOW = 51
LIDAR_RANGE = 20.0
NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
NUM_LOOKAHEAD = 10
OBS_FRENET_OFF = 3 + NUM_LIDAR
OBS_LOOK_OFF = OBS_FRENET_OFF + 2
OBS_DIM = OBS_LOOK_OFF + 2 * NUM_LOOKAHEAD
ACT_DIM = 2
MAX_STEPS = 10000
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

MU = 1.0489
FRONT_CORNERING_STIFFNESS = 4.718
REAR_CORNERING_STIFFNESS = 5.4562
LENGTH_FRONT = 0.15875
LENGTH_REAR = 0.17145
LENGTH_WHEELBASE = LENGTH_FRONT + LENGTH_REAR
CG_HEIGHT = 0.074
MASS = 3.74
INERTIA = 0.04712
STEER_MIN = -0.4189
STEER_MAX = 0.4189
STEER_V_MAX = 3.2
V_SWITCH = 7.319
V_BLEND_WIDTH = 0.5
V_BLEND_MIN = 1.0
A_MAX = 9.51
V_MIN = -5.0
V_MAX = 20.0
PSI_PRIME_MAX = 6.0
WIDTH = 0.31
LENGTH = 0.58
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0))
G = 9.81
DT = 1.0 / 60.0
SUBSTEPS = 6
DT_SUB = DT / float(SUBSTEPS)
DT_SUB_HALF = DT_SUB * 0.5
DT_SUB_SIX = DT_SUB / 6.0
DR_FRAC = 0.15
WALL_PENALTY_COEF = 0.1
WALL_PENALTY_RATE = 3.0
TERM_PENALTY = 100.0
SLIP_PENALTY_COEF = 0.1
PROGRESS_SCALE = 100.0
PROGRESS_V_COEF = 10.0

DONE_TERMINATED = 1
DONE_TRUNCATED = 2


@wp.struct
class VehicleD:
    d_x: float
    d_y: float
    d_psi: float
    d_beta: float
    dd_psi: float


@wp.func
def vehicle_dynamics_st(
    car_delta: float,
    car_v: float,
    car_psi: float,
    car_psi_prime: float,
    car_beta: float,
    steer_v: float,
    acceleration: float,
    mu_s: float,
    mass_s: float,
    lf_s: float,
    lr_s: float,
):
    state = VehicleD()

    lf = LENGTH_FRONT * lf_s
    lr = LENGTH_REAR * lr_s
    lwb = lf + lr
    inv_lwb = 1.0 / lwb
    mass = MASS * mass_s
    mu = MU * mu_s

    tand = wp.tan(car_delta)
    cosd = wp.cos(car_delta)
    cosd2 = cosd * cosd
    tl_ratio = tand * lr * inv_lwb
    inv_sq = 1.0 / wp.sqrt(1.0 + tl_ratio * tl_ratio)
    cbk = inv_sq
    sbk = tl_ratio * inv_sq

    cp = wp.cos(car_psi)
    sp = wp.sin(car_psi)
    d_x_k = car_v * (cbk * cp - sbk * sp)
    d_y_k = car_v * (sbk * cp + cbk * sp)
    d_psi_k = car_v * cbk * tand * inv_lwb
    d_beta_k = lr * steer_v * inv_lwb / cosd2 * inv_sq * inv_sq
    dd_psi_k = inv_lwb * (
        acceleration * cbk * tand
        - car_v * sbk * d_beta_k * tand
        + car_v * cbk * steer_v / cosd2
    )

    w_dyn = 0.5 * (wp.tanh((car_v - V_SWITCH) / V_BLEND_WIDTH) + 1.0)
    v_safe = wp.max(car_v, V_BLEND_MIN)
    inv_v_safe = 1.0 / v_safe

    g_lr_a = G * lr - acceleration * CG_HEIGHT
    g_lf_a = G * lf + acceleration * CG_HEIGHT
    cf = FRONT_CORNERING_STIFFNESS
    cr = REAR_CORNERING_STIFFNESS
    cf_glra = cf * g_lr_a
    cr_glfa = cr * g_lf_a
    lf_cf_glra = lf * cf_glra
    lr_cr_glfa = lr * cr_glfa

    mm_il = mu * mass * inv_lwb / INERTIA
    m_vl = mu * inv_lwb * inv_v_safe

    dd_psi_d = (
        -mm_il * inv_v_safe * (lf * lf_cf_glra + lr * lr_cr_glfa) * car_psi_prime
        + mm_il * (lr_cr_glfa - lf_cf_glra) * car_beta
        + mm_il * lf_cf_glra * car_delta
    )
    d_beta_d = (
        (m_vl * inv_v_safe * (lr_cr_glfa - lf_cf_glra) - 1.0) * car_psi_prime
        - m_vl * (cr_glfa + cf_glra) * car_beta
        + m_vl * cf_glra * car_delta
    )
    cb = wp.cos(car_beta)
    sb = wp.sin(car_beta)
    d_x_d = car_v * (cb * cp - sb * sp)
    d_y_d = car_v * (sb * cp + cb * sp)
    d_psi_d = car_psi_prime

    w_kin = 1.0 - w_dyn
    state.d_x = w_kin * d_x_k + w_dyn * d_x_d
    state.d_y = w_kin * d_y_k + w_dyn * d_y_d
    state.d_psi = w_kin * d_psi_k + w_dyn * d_psi_d
    state.dd_psi = w_kin * dd_psi_k + w_dyn * dd_psi_d
    state.d_beta = w_kin * d_beta_k + w_dyn * d_beta_d
    return state


@wp.func
def rk4_step(
    car_delta: float,
    car_v: float,
    car_psi: float,
    car_psi_prime: float,
    car_beta: float,
    steer_v: float,
    acceleration: float,
    mu_s: float,
    mass_s: float,
    lf_s: float,
    lr_s: float,
):
    dd_half = steer_v * DT_SUB_HALF
    dv_half = acceleration * DT_SUB_HALF
    dd_full = steer_v * DT_SUB
    dv_full = acceleration * DT_SUB

    k1 = vehicle_dynamics_st(
        car_delta,
        car_v,
        car_psi,
        car_psi_prime,
        car_beta,
        steer_v,
        acceleration,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k2 = vehicle_dynamics_st(
        car_delta + dd_half,
        car_v + dv_half,
        car_psi + k1.d_psi * DT_SUB_HALF,
        car_psi_prime + k1.dd_psi * DT_SUB_HALF,
        car_beta + k1.d_beta * DT_SUB_HALF,
        steer_v,
        acceleration,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k3 = vehicle_dynamics_st(
        car_delta + dd_half,
        car_v + dv_half,
        car_psi + k2.d_psi * DT_SUB_HALF,
        car_psi_prime + k2.dd_psi * DT_SUB_HALF,
        car_beta + k2.d_beta * DT_SUB_HALF,
        steer_v,
        acceleration,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k4 = vehicle_dynamics_st(
        car_delta + dd_full,
        car_v + dv_full,
        car_psi + k3.d_psi * DT_SUB,
        car_psi_prime + k3.dd_psi * DT_SUB,
        car_beta + k3.d_beta * DT_SUB,
        steer_v,
        acceleration,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    out = VehicleD()
    out.d_x = (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT_SUB_SIX
    out.d_y = (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT_SUB_SIX
    out.d_psi = (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT_SUB_SIX
    out.dd_psi = (
        k1.dd_psi + 2.0 * k2.dd_psi + 2.0 * k3.dd_psi + k4.dd_psi
    ) * DT_SUB_SIX
    out.d_beta = (
        k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta
    ) * DT_SUB_SIX
    return out


@wp.kernel
def step_kernel(
    actions: wp.array[wp.vec2],
    observation: wp.array2d[float],
    reward: wp.array[float],
    done: wp.array[int],
    cars: wp.array2d[float],
    cars_int: wp.array2d[int],
    car_dr: wp.array2d[float],
    origin: wp.vec2,
    res: float,
    distance_transform_px: wp.array2d[float],
    centerline_lut: wp.array2d[int],
    centerline: wp.array[wp.vec3],
    num_centerline_pts: int,
    look_step: int,
    lidar_dirs: wp.array[wp.vec2],
    seed_base: int,
    write_centerline: int,
):
    i = wp.tid()

    car_x = cars[i, 0]
    car_y = cars[i, 1]
    car_delta = cars[i, 2]
    car_v = cars[i, 3]
    car_psi = cars[i, 4]
    car_psi_prime = cars[i, 5]
    car_beta = cars[i, 6]
    car_steps = cars_int[i, 0]
    car_waypoint = cars_int[i, 1]

    mu_s = car_dr[i, 0]
    mass_s = car_dr[i, 1]
    lf_s = car_dr[i, 2]
    lr_s = car_dr[i, 3]

    origin_x = origin[0]
    origin_y = origin[1]
    map_w = distance_transform_px.shape[0]
    map_h = distance_transform_px.shape[1]
    map_h_f = wp.float32(map_h) - 1.0

    steer_v = wp.clamp(actions[i][0], -1.0, 1.0) * STEER_V_MAX
    if (steer_v < 0.0 and car_delta <= STEER_MIN) or (
        steer_v > 0.0 and car_delta >= STEER_MAX
    ):
        steer_v = 0.0
    acceleration = wp.clamp(actions[i][1], -1.0, 1.0) * A_MAX
    if (acceleration < 0.0 and car_v <= V_MIN) or (
        acceleration > 0.0 and car_v >= V_MAX
    ):
        acceleration = 0.0

    dd_sub = steer_v * DT_SUB
    dv_sub = acceleration * DT_SUB

    for _ in range(SUBSTEPS):
        d = rk4_step(
            car_delta,
            car_v,
            car_psi,
            car_psi_prime,
            car_beta,
            steer_v,
            acceleration,
            mu_s,
            mass_s,
            lf_s,
            lr_s,
        )
        car_x += d.d_x
        car_y += d.d_y
        car_delta += dd_sub
        car_v += dv_sub
        car_psi += d.d_psi
        car_psi_prime += d.dd_psi
        car_beta += d.d_beta

    car_delta = wp.clamp(car_delta, STEER_MIN, STEER_MAX)
    car_v = wp.clamp(car_v, V_MIN, V_MAX)
    car_psi_prime = wp.clamp(car_psi_prime, -PSI_PRIME_MAX, PSI_PRIME_MAX)
    car_beta = wp.clamp(car_beta, -1.2, 1.2)

    car_px = wp.clamp(wp.int32((car_x - origin_x) / res), 0, map_w - 1)
    car_py = wp.clamp(wp.int32(map_h_f - (car_y - origin_y) / res), 0, map_h - 1)

    edt_val = distance_transform_px[car_px, car_py] * res
    term = edt_val < CAR_HALF_DIAG
    trunc = car_steps >= MAX_STEPS
    car_steps += 1

    new_car_waypoint = centerline_lut[car_px, car_py]
    d_wp = new_car_waypoint - car_waypoint
    if 2 * d_wp > num_centerline_pts:
        d_wp -= num_centerline_pts
    elif 2 * d_wp < -num_centerline_pts:
        d_wp += num_centerline_pts

    progress = (
        wp.float32(d_wp)
        / wp.float32(num_centerline_pts)
        * PROGRESS_SCALE
        * (1.0 + wp.max(car_v, 0.0) / PROGRESS_V_COEF)
    )
    wall = -WALL_PENALTY_COEF * wp.exp(-WALL_PENALTY_RATE * edt_val)
    beta_kin = wp.atan(wp.tan(car_delta) * LENGTH_REAR / LENGTH_WHEELBASE)
    slip = -SLIP_PENALTY_COEF * (car_beta - beta_kin) * (car_beta - beta_kin)
    term_pen = wp.where(term, -TERM_PENALTY, 0.0)
    reward[i] = progress + wall + slip + term_pen

    if term:
        done[i] = DONE_TERMINATED
    elif trunc:
        done[i] = DONE_TRUNCATED
    else:
        done[i] = 0

    if trunc or term:
        rng = wp.rand_init(seed_base + i * 73 + car_steps * 31 + new_car_waypoint * 17)
        random_number = (
            wp.int32(wp.randf(rng) * wp.float32(num_centerline_pts))
            % num_centerline_pts
        )
        rpt = centerline[random_number]
        car_x = rpt[0]
        car_y = rpt[1]
        car_delta = 0.0
        car_v = 0.0
        car_psi = rpt[2]
        car_psi_prime = 0.0
        car_beta = 0.0
        car_steps = 0
        new_car_waypoint = random_number
        car_dr[i, 0] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 1] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 2] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 3] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)

    sh = wp.sin(car_psi)
    ch = wp.cos(car_psi)

    lidar_x = car_x + LENGTH_FRONT * ch
    lidar_y = car_y + LENGTH_FRONT * sh
    lidar_px = wp.clamp(wp.int32((lidar_x - origin_x) / res), 0, map_w - 1)
    lidar_py = wp.clamp(
        wp.int32(map_h_f - (lidar_y - origin_y) / res),
        0,
        map_h - 1,
    )
    lidar_pos_px = wp.vec2(wp.float32(lidar_px), wp.float32(lidar_py))
    lidar_range_px = LIDAR_RANGE / res

    for j in range(lidar_dirs.shape[0]):
        ray = lidar_pos_px
        dist_px = float(0.0)
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        d_px = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        while dist_px < lidar_range_px:
            ray_px = wp.int32(ray[0])
            ray_py = wp.int32(ray[1])
            if ray_px < 0 or ray_px >= map_w or ray_py < 0 or ray_py >= map_h:
                break
            dt_ray = distance_transform_px[ray_px, ray_py]
            ray += d_px * dt_ray
            dist_px += dt_ray
            if dt_ray == 0.0:
                break
        observation[i, j + 3] = wp.min(dist_px, lidar_range_px) * res

    if write_centerline != 0:
        cpt = centerline[new_car_waypoint]
        cx_pt = cpt[0]
        cy_pt = cpt[1]
        cth_pt = cpt[2]

        s_cth = wp.sin(cth_pt)
        c_cth = wp.cos(cth_pt)
        heading_err = wp.atan2(s_cth * ch - c_cth * sh, c_cth * ch + s_cth * sh)
        dx_c = car_x - cx_pt
        dy_c = car_y - cy_pt
        lateral_err = -dx_c * s_cth + dy_c * c_cth

        observation[i, OBS_FRENET_OFF] = heading_err
        observation[i, OBS_FRENET_OFF + 1] = lateral_err

        idx = new_car_waypoint
        for k in range(NUM_LOOKAHEAD):
            idx += look_step
            if idx >= num_centerline_pts:
                idx -= num_centerline_pts
            wpt = centerline[idx]
            dx = wpt[0] - car_x
            dy = wpt[1] - car_y
            observation[i, OBS_LOOK_OFF + k * 2] = dx * ch + dy * sh
            observation[i, OBS_LOOK_OFF + k * 2 + 1] = -dx * sh + dy * ch

    cars[i, 0] = car_x
    cars[i, 1] = car_y
    cars[i, 2] = car_delta
    cars[i, 3] = car_v
    cars[i, 4] = car_psi
    cars[i, 5] = car_psi_prime
    cars[i, 6] = car_beta
    cars_int[i, 0] = car_steps
    cars_int[i, 1] = new_car_waypoint

    observation[i, 0] = car_delta
    observation[i, 1] = car_v
    observation[i, 2] = car_psi_prime


class Map:
    def __init__(self, path: Path) -> None:
        self.meta = safe_load(path.read_text())
        img_path = path.parent / self.meta["image"]
        self.raw = imread(str(img_path), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(img_path)
        free = self.raw >= OCC_THRESH
        self.dt = distance_transform_edt(free)
        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])
        self._compute_centerline(free)
        self._build_lut()

    @staticmethod
    def _neighbors(skeleton, r, c, h, w):
        return [
            (r + dr, c + dc)
            for dr, dc in ADJ
            if 0 <= r + dr < h and 0 <= c + dc < w and skeleton[r + dr, c + dc]
        ]

    def _compute_centerline(self, free, smooth_window=SMOOTH_WINDOW):
        skeleton = skeletonize(free)
        h, w = skeleton.shape
        pts = np.argwhere(skeleton)
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start = tuple(int(x) for x in pts[np.argmin(((pts - origin_px) ** 2).sum(1))])
        nbrs = self._neighbors(skeleton, start[0], start[1], h, w)
        if len(nbrs) < 2:
            raise RuntimeError(
                f"Skeleton seed {start} has {len(nbrs)} neighbors; need a closed loop."
            )
        src, target = nbrs[0], nbrs[1]

        parent = {src: src}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(skeleton, r, c, h, w):
                n = (nr, nc)
                if n in parent or n == start:
                    continue
                parent[n] = (r, c)
                if n == target:
                    q.clear()
                    break
                q.append(n)
        path = [start]
        n = target
        while n != src:
            path.append(n)
            n = parent[n]
        path.append(src)
        path.reverse()
        rc = np.array(path)
        world = np.column_stack(
            [
                self.ox + rc[:, 1] * self.res,
                self.oy + (self.h - 1 - rc[:, 0]) * self.res,
            ]
        )
        self.centerline = savgol_filter(world, smooth_window, 3, axis=0, mode="wrap")
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        avg_sp = float(np.linalg.norm(diffs, axis=1).mean())
        self.look_step = max(1, int(round(1.0 / avg_sp)))

    def _build_lut(self):
        centerline_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        kdtree = KDTree(centerline_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        self.centerline_lut = kdtree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)


class RacingEnv:
    action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    def __init__(
        self,
        map_path: Path,
        num_envs: int,
        seed: int = 0,
        device: str | None = None,
        use_centerline_obs: bool = True,
    ):
        wp.init()
        self.num_envs = num_envs
        self.use_centerline_obs = use_centerline_obs
        self.obs_dim = OBS_DIM if use_centerline_obs else OBS_FRENET_OFF
        self._write_centerline = 1 if use_centerline_obs else 0
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_dim,), dtype=np.float32
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.map = Map(map_path)
        self.look_step = self.map.look_step
        self.seed_base = int(seed)
        d = self.device

        self.dt_buf = wp.array(self.map.dt.T.astype(np.float32), dtype=float, device=d)
        self.lut_buf = wp.array(
            self.map.centerline_lut.T.astype(np.int32), dtype=int, device=d
        )
        self.centerline_buf = wp.array(
            np.column_stack([self.map.centerline, self.map.angles]).astype(np.float32),
            dtype=wp.vec3,
            device=d,
        )
        self.num_centerline = len(self.map.centerline)

        rng = np.random.default_rng(seed)
        idxs = rng.integers(0, self.num_centerline, size=num_envs)
        cars = np.zeros((num_envs, 7), dtype=np.float32)
        cars[:, 0] = self.map.centerline[idxs, 0]
        cars[:, 1] = self.map.centerline[idxs, 1]
        cars[:, 4] = self.map.angles[idxs]
        cars_int = np.zeros((num_envs, 2), dtype=np.int32)
        cars_int[:, 1] = idxs

        dr_init = (
            1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((num_envs, 4), dtype=np.float32)
        )

        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.car_dr = wp.array(dr_init, dtype=float, device=d)
        self.obs = wp.zeros((num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(num_envs, dtype=float, device=d)
        self.done = wp.zeros(num_envs, dtype=int, device=d)
        self._wp_device = self.cars.device

        self.obs_buf = wp.to_torch(self.obs)
        self._obs_view = self.obs_buf[:, : self.obs_dim]
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self.cars_buf = wp.to_torch(self.cars)
        self._cars_int_buf = wp.to_torch(self.cars_int)
        self._car_dr_buf = wp.to_torch(self.car_dr)
        self._step_counter = self._cars_int_buf[:, 0]

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )

        self._zero_act = wp.zeros(num_envs, dtype=wp.vec2, device=d)
        self.nan_events = 0
        self._call_count = 0
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def _launch(self, act):
        seed = (self.seed_base * 2654435761 + self._call_count * 83492791) & 0x7FFFFFFF
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                act,
                self.obs,
                self.rew,
                self.done,
                self.cars,
                self.cars_int,
                self.car_dr,
                wp.vec2(self.map.ox, self.map.oy),
                self.map.res,
                self.dt_buf,
                self.lut_buf,
                self.centerline_buf,
                self.num_centerline,
                self.look_step,
                self.lidar_buf,
                int(seed),
                self._write_centerline,
            ],
        )
        wp.synchronize_device(self._wp_device)
        self._call_count += 1

    def _sanitize(self):
        bad = ~(
            torch.isfinite(self.obs_buf).all(1) & torch.isfinite(self.cars_buf).all(1)
        )
        if not bad.any().item():
            return
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self.nan_events += int(bad.sum().item())
        self._step_counter[bad] = MAX_STEPS
        self.done_buf[bad] = DONE_TRUNCATED

    def reset(self):
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self._obs_view, {}

    def step(self, actions):
        self._launch(wp.from_torch(actions.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        terminated = self.done_buf == DONE_TERMINATED
        truncated = self.done_buf == DONE_TRUNCATED
        return self._obs_view, self.rew_buf, terminated, truncated, {}

    def save_state(self):
        return {
            "cars": self.cars_buf.clone(),
            "cars_int": self._cars_int_buf.clone(),
            "car_dr": self._car_dr_buf.clone(),
            "obs": self.obs_buf.clone(),
            "rew": self.rew_buf.clone(),
            "done": self.done_buf.clone(),
        }

    def restore_state(self, s):
        self.cars_buf.copy_(s["cars"])
        self._cars_int_buf.copy_(s["cars_int"])
        self._car_dr_buf.copy_(s["car_dr"])
        self.obs_buf.copy_(s["obs"])
        self.rew_buf.copy_(s["rew"])
        self.done_buf.copy_(s["done"])


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape).float()
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta, alpha=bc / tot)
        m2 = self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)
        self.var = m2 / tot
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


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(
        self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256
    ):
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
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def _dist(self, obs):
        mean = self.actor(obs)
        log_std = self.actor_logstd.expand_as(mean).clamp(-1.6, -0.3)
        return Normal(mean, log_std.exp())

    def act_value(self, obs, action=None, need_entropy: bool = True):
        dist = self._dist(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1) if need_entropy else None
        value = self.value(obs)
        return action, log_prob, entropy, value

    def deterministic(self, obs):
        return self.actor(obs)


class KLAdaptiveLR:
    def __init__(
        self,
        optimizer,
        target_kl: float = 0.02,
        factor: float = 1.5,
        min_lr: float = 1e-6,
        max_lr: float = 3e-3,
    ):
        self.optimizer = optimizer
        self.target = target_kl
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr

    def step(self, kl: float) -> None:
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
            if kl > 2.0 * self.target:
                pg["lr"] = max(self.min_lr, lr / self.factor)
            elif kl < 0.5 * self.target:
                pg["lr"] = min(self.max_lr, lr * self.factor)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def record_rollout(
    env: RacingEnv,
    agent: Agent,
    num_steps: int,
    output_path: Path,
    obs_rms: RunningMeanStd | None = None,
    deterministic: bool = True,
) -> None:
    saved = env.save_state()
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

        trail: deque = deque(maxlen=300)
        raw_obs, _ = env.reset()
        obs = obs_rms.normalize(raw_obs) if obs_rms is not None else raw_obs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(output_path), fps=int(1 / DT), macro_block_size=1
        ) as writer:
            with torch.no_grad():
                for _ in range(num_steps):
                    action = (
                        agent.deterministic(obs)
                        if deterministic
                        else agent.act_value(obs, need_entropy=False)[0]
                    )
                    raw_obs, _, term, trunc, _ = env.step(action)
                    obs = obs_rms.normalize(raw_obs) if obs_rms is not None else raw_obs

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
                    writer.append_data(frame)
    finally:
        env.restore_state(saved)
        agent.train(was_training)


def _wandb_log(metrics, step, video_path=None):
    try:
        if video_path is not None:
            metrics = {
                **metrics,
                "rollout": wandb.Video(str(video_path), format="mp4"),
            }
        wandb.log(metrics, step=step)
    except Exception as exc:
        print(f"[warporacer] wandb log failed: {exc}")


def train(
    env: RacingEnv,
    agent: Agent,
    iterations: int = 2000,
    rollouts: int = 24,
    learning_epochs: int = 5,
    mini_batches: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    learning_rate: float = 3e-4,
    target_kl: float = 0.02,
    log_dir: Path = Path("./logs"),
    record_every: int = 100,
    record_steps: int = 1800,
):
    device = next(agent.parameters()).device
    num_envs = env.num_envs
    obs_dim = env.obs_dim
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    scheduler = KLAdaptiveLR(optimizer, target_kl=target_kl)

    obs_rms = RunningMeanStd((obs_dim,), device)
    ret_rms = ReturnNormalizer(num_envs, gamma, device)

    obs_buf = torch.zeros((rollouts, num_envs, obs_dim), device=device)
    act_buf = torch.zeros((rollouts, num_envs, ACT_DIM), device=device)
    logp_buf = torch.zeros((rollouts, num_envs), device=device)
    rew_buf = torch.zeros((rollouts, num_envs), device=device)
    done_buf = torch.zeros((rollouts, num_envs), device=device)
    term_buf = torch.zeros((rollouts, num_envs), device=device)
    val_buf = torch.zeros((rollouts, num_envs), device=device)
    ep_ret_buf = torch.zeros((rollouts, num_envs), device=device)
    ep_len_buf = torch.zeros((rollouts, num_envs), device=device)

    raw_obs, _ = env.reset()
    obs_rms.update(raw_obs)
    obs = obs_rms.normalize(raw_obs)

    ep_returns = torch.zeros(num_envs, device=device)
    ep_steps = torch.zeros(num_envs, device=device)
    completed_returns: deque = deque(maxlen=100)
    completed_lengths: deque = deque(maxlen=100)

    global_step = 0
    t0 = time.time()
    last_t = t0

    for it in range(iterations):
        agent.eval()
        with torch.no_grad():
            for t in range(rollouts):
                obs_buf[t] = obs
                action, log_prob, _, value = agent.act_value(obs, need_entropy=False)
                act_buf[t] = action
                logp_buf[t] = log_prob
                val_buf[t] = value

                raw_obs, raw_reward, term, trunc, _ = env.step(action)
                done = (term | trunc).float()

                ret_rms.update(raw_reward, done)
                rew_buf[t] = ret_rms.normalize(raw_reward)
                done_buf[t] = done
                term_buf[t] = term.float()

                ep_returns.add_(raw_reward)
                ep_steps.add_(1.0)
                ep_ret_buf[t] = ep_returns
                ep_len_buf[t] = ep_steps
                nonterm = 1.0 - done
                ep_returns.mul_(nonterm)
                ep_steps.mul_(nonterm)

                obs_rms.update(raw_obs)
                obs = obs_rms.normalize(raw_obs)

            next_value = agent.value(obs)
            val_ext = torch.cat([val_buf, next_value.unsqueeze(0)], 0)
            adv_buf = torch.zeros_like(rew_buf)
            last_gae = torch.zeros_like(next_value)
            for t in reversed(range(rollouts)):
                nonterm = 1.0 - term_buf[t]
                nondone = 1.0 - done_buf[t]
                delta = rew_buf[t] + gamma * val_ext[t + 1] * nonterm - val_buf[t]
                last_gae = delta + gamma * gae_lambda * nondone * last_gae
                adv_buf[t] = last_gae
            ret_buf = adv_buf + val_buf

        global_step += rollouts * num_envs

        finished = done_buf.bool()
        if finished.any():
            completed_returns.extend(ep_ret_buf[finished].cpu().tolist())
            completed_lengths.extend(ep_len_buf[finished].cpu().tolist())

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, ACT_DIM)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_val = val_buf.reshape(-1)
        batch_size = b_obs.shape[0]
        mb_size = batch_size // mini_batches

        agent.train()
        kl_t = torch.zeros((), device=device)
        clipfrac_t = torch.zeros((), device=device)
        pg_t = torch.zeros((), device=device)
        v_t = torch.zeros((), device=device)
        ent_t = torch.zeros((), device=device)
        n_updates = 0
        kl_stopped = False

        for epoch in range(learning_epochs):
            perm = torch.randperm(batch_size, device=device)
            epoch_kl = torch.zeros((), device=device)
            epoch_mb = 0
            for start in range(0, batch_size, mb_size):
                mb = perm[start : start + mb_size]
                _, new_logp, entropy, new_val = agent.act_value(b_obs[mb], b_act[mb])

                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                kl_t += approx_kl
                clipfrac_t += clipfrac
                epoch_kl += approx_kl
                epoch_mb += 1

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                mb_val = b_val[mb]
                mb_ret = b_ret[mb]
                v_err = new_val - mb_ret
                if vf_clip_coef > 0:
                    v_clipped = mb_val + (new_val - mb_val).clamp(
                        -vf_clip_coef, vf_clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            v_err.square(), (v_clipped - mb_ret).square()
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * v_err.square().mean()

                ent = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    agent.actor_logstd.clamp_(-1.6, -0.3)

                pg_t += pg_loss.detach()
                v_t += v_loss.detach()
                ent_t += ent.detach()
                n_updates += 1

            if (epoch_kl / float(epoch_mb)).item() > 1.5 * target_kl:
                kl_stopped = True
                break

        inv_n = 1.0 / max(n_updates, 1)
        mean_kl, mean_clipfrac, mean_pg, mean_v, mean_ent = (
            torch.stack([kl_t, clipfrac_t, pg_t, v_t, ent_t]) * inv_n
        ).tolist()
        scheduler.step(mean_kl)

        now = time.time()
        sps = int(rollouts * num_envs / max(now - last_t, 1e-9))
        last_t = now

        metrics = {
            "losses/policy_loss": mean_pg,
            "losses/value_loss": mean_v,
            "losses/entropy": mean_ent,
            "losses/approx_kl": mean_kl,
            "losses/clipfrac": mean_clipfrac,
            "losses/kl_early_stopped": int(kl_stopped),
            "policy/std": agent.actor_logstd.exp().mean().item(),
            "charts/learning_rate": scheduler.lr,
            "charts/sps": sps,
            "charts/iteration": it,
        }
        if completed_returns:
            metrics["charts/episodic_return"] = float(np.mean(completed_returns))
            metrics["charts/episodic_length"] = float(np.mean(completed_lengths))

        _wandb_log(metrics, global_step)
        if it % 10 == 0:
            er = metrics.get("charts/episodic_return", float("nan"))
            print(
                f"[iter {it:5d}] step={global_step:>10d} sps={sps:>6d} "
                f"ret={er:9.3f} kl={mean_kl:.4f} lr={scheduler.lr:.2e} "
                f"std={metrics['policy/std']:.3f}"
                f"{' KL-STOP' if kl_stopped else ''}"
            )

        if record_every > 0 and record_steps > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            try:
                record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
                _wandb_log({}, global_step, video_path=out)
            except Exception as exc:
                print(f"[warporacer] rollout at iter {it + 1} failed: {exc}")

    return time.time() - t0, obs_rms, ret_rms, global_step


def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("./logs"),
    device: str = "",
    record: int = 1800,
    record_every: int = 100,
    record_steps: int = 1800,
    use_wandb: bool = True,
    use_centerline_obs: bool = True,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env = RacingEnv(
        map_yaml,
        num_envs=num_envs,
        seed=seed,
        device=device or None,
        use_centerline_obs=use_centerline_obs,
    )
    agent = Agent(obs_dim=env.obs_dim).to(env.device)

    if use_wandb:
        try:
            wandb.init(
                project="warporacer",
                name=f"seed{seed}_n{num_envs}",
                config={
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "seed": seed,
                    "map": str(map_yaml),
                    "obs_dim": env.obs_dim,
                    "use_centerline_obs": use_centerline_obs,
                    "act_dim": ACT_DIM,
                    "substeps": SUBSTEPS,
                    "dr_frac": DR_FRAC,
                    "num_lookahead": NUM_LOOKAHEAD,
                },
            )
        except Exception as exc:
            print(f"[warporacer] wandb init failed: {exc}")

    elapsed, obs_rms, ret_rms, global_step = train(
        env,
        agent,
        iterations=iterations,
        log_dir=log_dir,
        record_every=record_every,
        record_steps=record_steps,
    )
    print(
        f"[warporacer] training finished in {elapsed:.1f}s "
        f"({env.nan_events} sanitized NaN events)"
    )

    torch.save(
        {
            "agent": agent.state_dict(),
            "obs_rms_mean": obs_rms.mean.cpu(),
            "obs_rms_var": obs_rms.var.cpu(),
            "obs_rms_count": obs_rms.count,
            "ret_rms_var": ret_rms.rms.var.cpu(),
            "ret_rms_count": ret_rms.rms.count,
        },
        log_dir / "agent_final.pt",
    )

    if record > 0:
        out = log_dir / "rollout.mp4"
        record_rollout(env, agent, record, out, obs_rms=obs_rms)
        _wandb_log({}, global_step, video_path=out)


if __name__ == "__main__":
    run(main)
