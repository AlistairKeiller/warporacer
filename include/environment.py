from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
import warp as wp

from include.constants import *
from include.map import Map
from include.warped_functions import step_kernel


class Environment:
    # Class attribute type annotations
    map_yaml: Path
    num_envs: int
    seed: int
    seed_base: int
    device: str
    map: Map
    vs: Visuals
    
    # Warp array buffers
    dt_buf: wp.array
    lut_buf: wp.array
    centerline_buf: wp.array
    cars: wp.array
    cars_int: wp.array
    car_dr: wp.array
    obs: wp.array
    rew: wp.array
    done: wp.array
    lidar_buf: wp.array
    _zero_act: wp.array
    
    # Pre-allocated constant structural types
    map_origin: wp.vec2
    
    # Interop PyTorch tensor views
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    done_buf: torch.Tensor
    cars_buf: torch.Tensor
    cars_int_buf: torch.Tensor
    _step_counter: torch.Tensor
    
    # Pre-allocated optimization buffers
    term_buf: torch.Tensor
    trunc_buf: torch.Tensor
    _empty_info: Dict[str, Any]
    
    # Scalar trackers
    n_cl: int
    look_step: int
    _call: int

    def __init__(self,
                 map_yaml: Path = Path(".\\maps\\berlin.yaml"),
                 num_envs: int = 1,
                 seed: int = 0,
                 live_viewer: bool = True
        ) -> None:
        self.map_yaml = map_yaml
        self.num_envs = num_envs
        self.seed = seed
        self.seed_base = seed 
        self.live_viewer = live_viewer

        self.device = wp.get_device()
        self.map = Map(self.map_yaml)

        # Initialize core physics tracking variables
        self._init_cars()

        # LAZY IMPORT: Keeps visuals.py (and pyglet) completely dormant unless requested
        if self.live_viewer:
            from include.visuals import Visuals
            self.vs = Visuals(self, self.map)
        else:
            self.vs = None

    def _init_cars(self) -> None:
        self.look_step = self.map.look_step
        d: str = self.device

        # Transfer track map data structures to Warp device arrays
        self.dt_buf = wp.array(self.map.dt.T.astype(np.float32), dtype=float, device=d)
        self.lut_buf = wp.array(self.map.lut.T.astype(np.int32), dtype=int, device=d)
        self.centerline_buf = wp.array(
            np.column_stack([self.map.centerline, self.map.angles]).astype(np.float32),
            dtype=wp.vec3,
            device=d,
        )
        self.n_cl = len(self.map.centerline)

        # Pre-allocate static map properties to avoid allocation inside launch loop
        self.map_origin = wp.vec2(self.map.ox, self.map.oy)

        # Sample initial spawn points using a randomized generator
        rng: np.random.Generator = np.random.default_rng(self.seed)
        idxs: np.ndarray = rng.integers(0, self.n_cl, size=self.num_envs)
        
        # State representations: x, y, x_vel, y_vel, theta, angular_vel, steer
        cars: np.ndarray = np.zeros((self.num_envs, 7), dtype=np.float32)
        cars[:, 0] = self.map.centerline[idxs, 0]
        cars[:, 1] = self.map.centerline[idxs, 1]
        cars[:, 4] = self.map.angles[idxs]
        
        cars_int: np.ndarray = np.zeros((self.num_envs, 2), dtype=np.int32)
        cars_int[:, 1] = idxs
        
        dr_init: np.ndarray = (
            1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 4), dtype=np.float32)
        )

        # Instantiate native Warp arrays on the designated compute device
        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.car_dr = wp.array(dr_init, dtype=float, device=d)
        self.obs = wp.zeros((self.num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(self.num_envs, dtype=float, device=d)
        self.done = wp.zeros(self.num_envs, dtype=int, device=d)

        # Expose zero-copy PyTorch tensor views sharing the underlying allocations
        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self.cars_buf = wp.to_torch(self.cars)
        self.cars_int_buf = wp.to_torch(self.cars_int)
        self._step_counter = self.cars_int_buf[:, 0]

        # Pre-allocate static masks to eliminate allocations in the step loop
        self.term_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.obs_buf.device)
        self.trunc_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.obs_buf.device)
        self._empty_info = {}

        # Compute fixed relative angles for the onboard Lidar array
        angles: np.ndarray = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )
        self._zero_act = wp.zeros(self.num_envs, dtype=wp.vec2, device=d)
        self._call = 0

        # Execute warm-up sequence to force lazy initialization components
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Cast PyTorch action tensors directly to Warp vectors and step the physics
        self._launch(wp.from_torch(action.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        
        # Write termination states into pre-allocated memory masks
        torch.eq(self.done_buf, DONE_TERMINATED, out=self.term_buf)
        torch.eq(self.done_buf, DONE_TRUNCATED, out=self.trunc_buf)

        return (
            self.obs_buf,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self._empty_info,
        )
    
    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._step_counter.fill_(MAX_STEPS)
        self._launch(self._zero_act)
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self.obs_buf, self._empty_info

    def save_state(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k).clone()
            for k in ("cars_buf", "cars_int_buf", "obs_buf", "rew_buf", "done_buf")
        } | {
            "car_dr": wp.to_torch(self.car_dr).clone(),
        }

    def restore_state(self, s: Dict[str, Any]) -> None:
        self.cars_buf.copy_(s["cars_buf"])
        self.cars_int_buf.copy_(s["cars_int_buf"])
        wp.to_torch(self.car_dr).copy_(s["car_dr"])
        self.obs_buf.copy_(s["obs_buf"])
        self.rew_buf.copy_(s["rew_buf"])
        self.done_buf.copy_(s["done_buf"])

    def _launch(self, act: wp.array) -> None:
        # Linear congruential generation style sequence for deterministic pseudo-random seeds
        seed: int = (self.seed_base * 2654435761 + self._call * 83492791) & 0x7FFFFFFF
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
                self.map_origin,  # Passed pre-allocated structural object reference
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
        self._call += 1
    
    def _sanitize(self) -> None:
        # Globally repair any numerical instability markers in-place
        # (The kernel now sets the 'done' flags for these automatically on-device)
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)