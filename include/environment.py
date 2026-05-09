from pathlib import Path

import numpy as np
import torch
import warp as wp

from include.constants import *
from include.map import Map
from include.visuals import Visuals
from include.warped_functions import step_kernel


class Environment:
    def __init__(self,
                 map_yaml: Path = Path(".\\maps\\berlin.yaml"),
                 num_envs: int = 1,
                 seed: int = 0
        ):
        self.map_yaml = map_yaml
        self.num_envs = num_envs
        self.seed = seed
        self.seed_base = seed # Why two versions?

        self.device = wp.get_device()
        self.map = Map(self.map_yaml)

        self._init_cars()

        self.vs = Visuals(self, self.map)

    def _init_cars(self):
        self.look_step = self.map.look_step
        d = self.device

        self.dt_buf = wp.array(self.map.dt.T.astype(np.float32), dtype=float, device=d)
        self.lut_buf = wp.array(self.map.lut.T.astype(np.int32), dtype=int, device=d)
        self.centerline_buf = wp.array(
            np.column_stack([self.map.centerline, self.map.angles]).astype(np.float32),
            dtype=wp.vec3,
            device=d,
        )
        self.n_cl = len(self.map.centerline)

        rng = np.random.default_rng(self.seed)
        idxs = rng.integers(0, self.n_cl, size=self.num_envs)
        cars = np.zeros((self.num_envs, 7), dtype=np.float32)
        cars[:, 0] = self.map.centerline[idxs, 0]
        cars[:, 1] = self.map.centerline[idxs, 1]
        cars[:, 4] = self.map.angles[idxs]
        cars_int = np.zeros((self.num_envs, 2), dtype=np.int32)
        cars_int[:, 1] = idxs
        dr_init = (
            1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 4), dtype=np.float32)
        )

        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.car_dr = wp.array(dr_init, dtype=float, device=d)
        self.obs = wp.zeros((self.num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(self.num_envs, dtype=float, device=d)
        self.done = wp.zeros(self.num_envs, dtype=int, device=d)

        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self.cars_buf = wp.to_torch(self.cars)
        self.cars_int_buf = wp.to_torch(self.cars_int)
        self._step_counter = self.cars_int_buf[:, 0]

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )
        self._zero_act = wp.zeros(self.num_envs, dtype=wp.vec2, device=d)
        self._call = 0

        # Warm-up reset
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def step(self, action):
        self._launch(wp.from_torch(action.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        return (
            self.obs_buf,
            self.rew_buf,
            self.done_buf == DONE_TERMINATED,
            self.done_buf == DONE_TRUNCATED,
            {},
        )
    
    def reset(self):
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self.obs_buf, {}

    def save_state(self):
        return {
            k: getattr(self, k).clone()
            for k in ("cars_buf", "cars_int_buf", "obs_buf", "rew_buf", "done_buf")
        } | {
            "car_dr": wp.to_torch(self.car_dr).clone(),
        }

    def restore_state(self, s):
        self.cars_buf.copy_(s["cars_buf"])
        self.cars_int_buf.copy_(s["cars_int_buf"])
        wp.to_torch(self.car_dr).copy_(s["car_dr"])
        self.obs_buf.copy_(s["obs_buf"])
        self.rew_buf.copy_(s["rew_buf"])
        self.done_buf.copy_(s["done_buf"])

    def _launch(self, act):
        seed = (self.seed_base * 2654435761 + self._call * 83492791) & 0x7FFFFFFF
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
                self.n_cl,
                self.look_step,
                self.lidar_buf,
                int(seed),
            ],
        )
        wp.synchronize_device(self.cars.device)
        self._call += 1
    
    def _sanitize(self):
        bad = ~(
            torch.isfinite(self.obs_buf).all(1) & torch.isfinite(self.cars_buf).all(1)
        )
        if not bad.any():
            return
        
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)

        self._step_counter[bad] = MAX_STEPS
        self.done_buf[bad] = DONE_TRUNCATED