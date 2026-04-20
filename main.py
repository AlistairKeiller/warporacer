# pyright: reportIndexIssue=false
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import warp as wp
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import Wrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from typer import run
from yaml import safe_load

OCC_THRESH = 230
SMOOTH_WINDOW = 51
LIDAR_RANGE = 20.0
NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
OBS_DIM = 3 + NUM_LIDAR
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
A_MAX = 9.51
V_MIN = -5.0
V_MAX = 20.0
WIDTH = 0.31
LENGTH = 0.58
G = 9.81
DT = 1 / 60


@wp.struct
class VehicleD:
    d_x: float
    d_y: float
    d_delta: float
    d_v: float
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
):
    state = VehicleD()
    state.d_delta = steer_v
    state.d_v = acceleration
    if wp.abs(car_v) < 0.5:
        beta = wp.atan(wp.tan(car_delta) * LENGTH_REAR / LENGTH_WHEELBASE)
        state.d_x = car_v * wp.cos(beta + car_psi)
        state.d_y = car_v * wp.sin(beta + car_psi)
        state.d_psi = car_v * wp.cos(beta) * wp.tan(car_delta) / LENGTH_WHEELBASE
        state.d_beta = (LENGTH_REAR * steer_v) / (
            LENGTH_WHEELBASE
            * wp.cos(car_delta) ** 2.0
            * (1.0 + (wp.tan(car_delta) * LENGTH_REAR / LENGTH_WHEELBASE) ** 2.0)
        )
        state.dd_psi = (
            1.0
            / LENGTH_WHEELBASE
            * (
                acceleration * wp.cos(car_beta) * wp.tan(car_delta)
                - car_v * wp.sin(car_beta) * state.d_beta * wp.tan(car_delta)
                + car_v * wp.cos(car_beta) * steer_v / wp.cos(car_delta) ** 2.0
            )
        )
    else:
        state.d_x = car_v * wp.cos(car_beta + car_psi)
        state.d_y = car_v * wp.sin(car_beta + car_psi)
        state.d_psi = car_psi_prime
        state.dd_psi = (
            -MU
            * MASS
            / (car_v * INERTIA * LENGTH_WHEELBASE)
            * (
                LENGTH_FRONT**2.0
                * FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
                + LENGTH_REAR**2.0
                * REAR_CORNERING_STIFFNESS
                * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
            )
            * car_psi_prime
            + MU
            * MASS
            / (INERTIA * LENGTH_WHEELBASE)
            * (
                LENGTH_REAR
                * REAR_CORNERING_STIFFNESS
                * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                - LENGTH_FRONT
                * FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
            )
            * car_beta
            + MU
            * MASS
            / (INERTIA * LENGTH_WHEELBASE)
            * LENGTH_FRONT
            * FRONT_CORNERING_STIFFNESS
            * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
            * car_delta
        )
        state.d_beta = (
            (
                MU
                / (car_v**2.0 * LENGTH_WHEELBASE)
                * (
                    REAR_CORNERING_STIFFNESS
                    * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                    * LENGTH_REAR
                    - FRONT_CORNERING_STIFFNESS
                    * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
                    * LENGTH_FRONT
                )
                - 1.0
            )
            * car_psi_prime
            - MU
            / (car_v * LENGTH_WHEELBASE)
            * (
                REAR_CORNERING_STIFFNESS * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                + FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
            )
            * car_beta
            + MU
            / (car_v * LENGTH_WHEELBASE)
            * (FRONT_CORNERING_STIFFNESS * (G * LENGTH_REAR - acceleration * CG_HEIGHT))
            * car_delta
        )
    return state


@wp.kernel
def step(
    actions: wp.array[wp.vec2],
    observation: wp.array2d[float],
    reward: wp.array[float],
    cars: wp.array2d[float],
    cars_int: wp.array2d[int],
    origin: wp.vec2,
    res: float,
    distance_transform_px: wp.array2d[float],
    centerline_lut: wp.array2d[int],
    centerline: wp.array[wp.vec3],
    num_centerline_pts: int,
    lidar_dirs: wp.array[wp.vec2],
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

    origin_x = origin[0]
    origin_y = origin[1]

    map_w = distance_transform_px.shape[0]
    map_h = distance_transform_px.shape[1]

    steer_action = actions[i][0]
    acceleration_action = actions[i][1]

    steer_v = wp.clamp(steer_action, -1.0, 1.0) * STEER_V_MAX
    if steer_v < 0 and car_delta <= STEER_MIN or steer_v > 0 and car_delta >= STEER_MAX:
        steer_v = 0.0
    acceleration = wp.clamp(acceleration_action, -1.0, 1.0) * A_MAX
    if acceleration < 0 and car_v <= V_MIN or acceleration > 0 and car_v >= V_MAX:
        acceleration = 0.0

    k1 = vehicle_dynamics_st(
        car_delta, car_v, car_psi, car_psi_prime, car_beta, steer_v, acceleration
    )
    k2 = vehicle_dynamics_st(
        car_delta + k1.d_delta * DT * 0.5,
        car_v + k1.d_v * DT * 0.5,
        car_psi + k1.d_psi * DT * 0.5,
        car_psi_prime + k1.dd_psi * DT * 0.5,
        car_beta + k1.d_beta * DT * 0.5,
        steer_v,
        acceleration,
    )
    k3 = vehicle_dynamics_st(
        car_delta + k2.d_delta * DT * 0.5,
        car_v + k2.d_v * DT * 0.5,
        car_psi + k2.d_psi * DT * 0.5,
        car_psi_prime + k2.dd_psi * DT * 0.5,
        car_beta + k2.d_beta * DT * 0.5,
        steer_v,
        acceleration,
    )
    k4 = vehicle_dynamics_st(
        car_delta + k3.d_delta * DT,
        car_v + k3.d_v * DT,
        car_psi + k3.d_psi * DT,
        car_psi_prime + k3.dd_psi * DT,
        car_beta + k3.d_beta * DT,
        steer_v,
        acceleration,
    )

    car_x += (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT / 6.0
    car_y += (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT / 6.0
    car_delta += (
        (k1.d_delta + 2.0 * k2.d_delta + 2.0 * k3.d_delta + k4.d_delta) * DT / 6.0
    )
    car_v += (k1.d_v + 2.0 * k2.d_v + 2.0 * k3.d_v + k4.d_v) * DT / 6.0
    car_psi += (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT / 6.0
    car_psi_prime += (
        (k1.dd_psi + 2.0 * k2.dd_psi + 2.0 * k3.dd_psi + k4.dd_psi) * DT / 6.0
    )
    car_beta += (k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta) * DT / 6.0

    car_delta = wp.clamp(car_delta, STEER_MIN, STEER_MAX)
    car_v = wp.clamp(car_v, V_MIN, V_MAX)

    car_px = wp.clamp(wp.int32((car_x - origin_x) / res), 0, map_w - 1)
    car_py = wp.clamp(
        wp.int32(float(map_h) - 1.0 - (car_y - origin_y) / res), 0, map_h - 1
    )

    car_pos_px = wp.vec2(wp.float32(car_px), wp.float32(car_py))

    term = distance_transform_px[car_px, car_py] * res < wp.length(
        wp.vec2(WIDTH / 2.0, LENGTH / 2.0)
    )
    trunc = car_steps >= MAX_STEPS
    car_steps += 1

    new_car_waypoint = centerline_lut[car_px, car_py]
    d_centerline_pt = new_car_waypoint - car_waypoint
    if d_centerline_pt > num_centerline_pts / 2:
        d_centerline_pt -= num_centerline_pts
    elif d_centerline_pt < -num_centerline_pts / 2:
        d_centerline_pt += num_centerline_pts
    reward[i] = wp.float32(d_centerline_pt) / wp.float32(
        num_centerline_pts
    ) - wp.float32(term)

    if trunc or term:
        seed = i * 2654435761 + new_car_waypoint * 2246822519 + car_steps * 3266489917
        random_number = wp.int32(wp.uint32(seed) >> wp.uint32(16)) % num_centerline_pts
        car_x = centerline[random_number][0]
        car_y = centerline[random_number][1]
        car_delta = 0.0
        car_v = 0.0
        car_psi = centerline[random_number][2]
        car_psi_prime = 0.0
        car_beta = 0.0
        car_steps = 0
        new_car_waypoint = random_number

        car_px = wp.clamp(wp.int32((car_x - origin_x) / res), 0, map_w - 1)
        car_py = wp.clamp(
            wp.int32(float(map_h) - 1.0 - (car_y - origin_y) / res), 0, map_h - 1
        )
        car_pos_px = wp.vec2(wp.float32(car_px), wp.float32(car_py))

    sh, ch = wp.sin(car_psi), wp.cos(car_psi)
    for j in range(lidar_dirs.shape[0]):
        ray = wp.vec2(wp.float32(car_px), wp.float32(car_py))
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        d_px = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        while wp.length(ray - car_pos_px) * res < LIDAR_RANGE:
            ray_px = wp.int32(ray[0])
            ray_py = wp.int32(ray[1])
            if ray_px < 0 or ray_px >= map_w or ray_py < 0 or ray_py >= map_h:
                break
            dt_ray = distance_transform_px[ray_px, ray_py]
            ray += d_px * dt_ray
            if dt_ray == 0.0:
                break
        observation[i, j + 3] = wp.min(wp.length(ray - car_pos_px) * res, LIDAR_RANGE)

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
        with open(path, "r") as f:
            self.meta = safe_load(f)
        self.raw = imread(str(path.parent / self.meta["image"]), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(path.parent / self.meta["image"])
        self.dt = distance_transform_edt(self.raw >= OCC_THRESH)
        self.ox, self.oy, self.ophi = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])
        self._compute_centerline(self.raw)
        self._build_lut()

    def _compute_centerline(self, raw, smooth_window=SMOOTH_WINDOW):
        skeleton = skeletonize(raw >= OCC_THRESH)

        pts = np.argwhere(skeleton)
        origin_px = [self.h - 1 + self.oy / self.res, -self.ox / self.res]
        start = tuple(pts[np.argmin(np.linalg.norm(pts - origin_px, axis=1))])

        nbrs = [
            (start[0] + dr, start[1] + dc)
            for dr, dc in ADJ
            if skeleton[start[0] + dr, start[1] + dc]
        ]
        src, target = nbrs[0], nbrs[1]
        parent = {src: src}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for dr, dc in ADJ:
                n = (r + dr, c + dc)
                if skeleton[n] and n not in parent and n != start:
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
        self.diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(self.diffs[:, 1], self.diffs[:, 0])

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
    def __init__(
        self, map_path: Path, num_envs: int, seed: int = 0, device: str | None = None
    ):
        wp.init()
        self.num_envs = num_envs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        self.map = Map(map_path)
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

        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.obs = wp.zeros((num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(num_envs, dtype=float, device=d)

        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self._cars_buf = wp.to_torch(self.cars)
        self._step_counter = wp.to_torch(self.cars_int)[:, 0]

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR).astype(
            np.float32
        )
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )

        self._launch(wp.zeros(num_envs, dtype=wp.vec2, device=d))
        self._sanitize()

    @property
    def unwrapped(self):
        return self

    def _launch(self, act):
        wp.launch(
            step,
            dim=self.num_envs,
            inputs=[
                act,
                self.obs,
                self.rew,
                self.cars,
                self.cars_int,
                wp.vec2(self.map.ox, self.map.oy),
                self.map.res,
                self.dt_buf,
                self.lut_buf,
                self.centerline_buf,
                self.num_centerline,
                self.lidar_buf,
            ],
        )
        wp.synchronize()

    def _sanitize(self):
        nan_mask = torch.isnan(self._cars_buf).any(dim=1) | torch.isinf(
            self._cars_buf
        ).any(dim=1)
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self._cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        if nan_mask.any():
            self._step_counter[nan_mask] = MAX_STEPS

    def reset(self, seed=None, options=None):
        return self.obs_buf, {}

    def step(self, actions):
        self._launch(wp.from_torch(actions.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        terminated = (self._step_counter == 0).unsqueeze(-1)
        truncated = torch.zeros_like(terminated)
        return (
            self.obs_buf,
            self.rew_buf.unsqueeze(-1),
            terminated,
            truncated,
            {},
        )

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass


class WarpEnvWrapper(Wrapper):
    """Thin skrl wrapper around RacingEnv."""

    def __init__(self, env: RacingEnv):
        super().__init__(env)
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._state_space = env.observation_space
        self._num_envs = env.num_envs
        self._num_agents = 1
        self._device = torch.device(env.device)

    def step(self, actions):
        return self._env.step(actions)

    def reset(self):
        return self._env.reset()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("./logs"),
    device: str = "",
):
    set_seed(seed)

    raw_env = RacingEnv(map_yaml, num_envs=num_envs, seed=seed, device=device or None)
    env = WarpEnvWrapper(raw_env)

    ROLLOUTS = 24

    memory = RandomMemory(
        memory_size=ROLLOUTS, num_envs=env.num_envs, device=env.device
    )

    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device),
    }

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = ROLLOUTS
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.005
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0  # disable hard KL early-stop; scheduler handles LR
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    cfg["experiment"]["directory"] = str(log_dir)
    cfg["experiment"]["experiment_name"] = "warporacer"
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 1000

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    trainer_cfg = {"timesteps": iterations * ROLLOUTS, "headless": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()


if __name__ == "__main__":
    run(main)
