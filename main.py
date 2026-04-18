# pyright: reportIndexIssue=false
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from typer import run
from yaml import safe_load

OCC_THRESH = 230
SMOOTH_WINDOW = 51
LIDAR_RANGE = 20.0
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
STEER_V_MIN = -3.2
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
    # kenematic model
    if wp.abs(car_v) < 0.1:
        beta = wp.atan(wp.tan(car_delta) * LENGTH_REAR / LENGTH_WHEELBASE)
        state.d_x = car_v * wp.cos(beta + car_psi)
        state.d_y = car_v * wp.sin(beta + car_psi)
        state.d_psi = car_v * wp.cos(beta) * wp.tan(car_delta) / LENGTH_WHEELBASE
        state.d_beta = (LENGTH_REAR * steer_v) / (
            LENGTH_WHEELBASE
            * wp.cos(car_delta) ** 2
            * (1 + (wp.tan(car_delta) ** 2 * LENGTH_REAR / LENGTH_WHEELBASE) ** 2)
        )
        state.dd_psi = (
            1
            / LENGTH_WHEELBASE
            * (
                acceleration * wp.cos(car_beta) * wp.tan(car_delta)
                - car_v * wp.sin(car_beta) * state.d_beta * wp.tan(car_delta)
                + car_v * wp.cos(car_beta) * steer_v / wp.cos(car_delta) ** 2
            )
        )
    # dynamic model
    else:
        state.d_x = car_v * wp.cos(car_beta + car_psi)
        state.d_y = car_v * wp.sin(car_beta + car_psi)
        state.d_delta = steer_v
        state.d_beta = (
            -MU
            * MASS
            / (car_v * INERTIA * LENGTH_WHEELBASE)
            * (
                LENGTH_FRONT**2
                * FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
                + LENGTH_REAR**2
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
        state.dd_psi = (
            (
                MU
                / (car_v**2 * LENGTH_WHEELBASE)
                * (
                    REAR_CORNERING_STIFFNESS
                    * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                    * LENGTH_REAR
                    - FRONT_CORNERING_STIFFNESS
                    * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
                    * LENGTH_FRONT
                )
                - 1
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
    cars: wp.array2d[float],
    actions: wp.array[wp.vec2],
    origin: wp.vec2,
    res: float,
    dt_px: wp.array2d[float],
    lidar_dirs: wp.array[wp.vec2],
    lidar_ranges: wp.array2d[float],
):
    i = wp.tid()

    car_x = cars[i, 0]
    car_y = cars[i, 1]
    car_delta = cars[i, 2]
    car_v = cars[i, 3]
    car_psi = cars[i, 4]
    car_psi_prime = cars[i, 5]
    car_beta = cars[i, 6]
    car_step = cars[i, 7]
    car_centerline_pt = cars[i, 8]

    origin_x = origin[0]
    origin_y = origin[1]

    # width_px = dt_px.size[0]
    height_px = dt_px.size[1]

    car_px = (car_x - origin_x) / res
    car_py = height_px - 1 - (car_y - origin_y) / res

    # step logic
    steer_v = wp.clamp(actions[i, 0] * STEER_V_MAX, STEER_V_MIN, STEER_V_MAX)
    if steer_v < 0 and car_delta <= STEER_MIN or steer_v > 0 and car_delta >= STEER_MAX:
        steer_v = 0
    acceleration = wp.clamp(actions[i, 1], -1, 1) * A_MAX
    if acceleration < 0 and car_v <= V_MIN or acceleration > 0 and car_v >= V_MAX:
        acceleration = 0

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

    cars[i, 0] += (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT / 6.0
    cars[i, 1] += (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT / 6.0
    cars[i, 2] += (
        (k1.d_delta + 2.0 * k2.d_delta + 2.0 * k3.d_delta + k4.d_delta) * DT / 6.0
    )
    cars[i, 3] += (k1.d_v + 2.0 * k2.d_v + 2.0 * k3.d_v + k4.d_v) * DT / 6.0
    cars[i, 4] += (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT / 6.0
    cars[i, 5] += (k1.dd_psi + 2.0 * k2.dd_psi + 2.0 * k3.dd_psi + k4.dd_psi) * DT / 6.0
    cars[i, 6] += (k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta) * DT / 6.0

    # collision logic
    term = dt_px[car_px, car_py] * res < wp.length(wp.vec2([WIDTH / 2, LENGTH / 2]))
    trunc = car_step >= MAX_STEPS
    cars[i, 7] += 1

    # reward logic
    new_centerline_pt =

# def car_colides(
#     car: wp.vec3, origin: wp.vec2, res: float, dt: wp.array2d[float]
# ) -> bool:
#     car_px = (car[0] - origin[0]) / res
#     car_py = dt.size[1] - 1 - (car[1] - origin[1]) / res


# @wp.kernel
# def scan_lidars(
#     cars: wp.array[wp.vec3],
#     origin: wp.vec2,
#     res: float,
#     dt: wp.array2d[float],
#     lidar_dirs: wp.array[wp.vec2],
#     max_distance: float,
#     lidar_ranges: wp.array2d[float],
# ):
#     i = wp.tid()
#     car = cars[i]
#     car_px = (car[0] - origin[0]) / res
#     car_py = dt.size[1] - 1 - (car[1] - origin[1]) / res
#     car_phi = car[2]
#     ray = wp.vec2(car_px, car_py)
#     sh, ch = wp.sin(car_phi), wp.cos(car_phi)
#     for j in range(len(lidar_dirs)):
#         ca, sa = lidar_dirs[j]
#         d_px = wp.vec2(ch * ca - sh * sa, sh * ca + ch * sa)
#         while wp.length(ray - wp.vec2(car_px, car_py)) < max_distance:
#             ray_px = wp.int32(ray[0])
#             ray_py = wp.int32(ray[1])
#             dt_ray = dt[ray_px, ray_py]
#             ray += d_px * dt_ray
#             if dt_ray == 0.0:
#                 lidar_ranges[i, j] = wp.length(ray - wp.vec2(car_px, car_py))
#                 break


class Map:
    def __init__(self, path: Path) -> None:
        with open(path, "r") as f:
            self.meta = safe_load(f)
        raw = imread(str(path.parent / self.meta["image"]), IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(path.parent / self.meta["image"])
        self.occupied = wp.array(raw < OCC_THRESH)
        self.dt = wp.array(distance_transform_edt(raw >= OCC_THRESH))
        self.ox, self.oy, self.ophi = self.meta["origin"]
        self.h, self.w = self.occupied.shape
        self.res = float(self.meta["resolution"])
        self._compute_centerline(raw)

    def _compute_centerline(self, raw, smooth_window=SMOOTH_WINDOW):
        skeleton = skeletonize(raw >= OCC_THRESH)
        skeleton_points = np.argwhere(skeleton)

        def neighbors(p):
            return (
                (p[0] + d[0], p[1] + d[1])
                for d in ADJ
                if 0 <= p[0] + d[0] < self.h
                and 0 <= p[1] + d[1] < self.w
                and skeleton[p[0] + d[0], p[1] + d[1]]
            )

        origin_row = self.h - 1 + self.oy / self.res
        origin_col = -self.ox / self.res
        start = np.argmin(
            np.linalg.norm(skeleton_points - np.array((origin_row, origin_col)), axis=1)
        )
        starting_neighbors = list(neighbors(skeleton_points[start]))
        src, targets = starting_neighbors[0], starting_neighbors[1:]
        parent = {src: src}
        q = deque([src])
        found = None
        while q and found is None:
            cur = q.popleft()
            for n in neighbors(cur):
                if n not in parent and n != start:
                    parent[n] = cur
                    q.append(n)
                    if n in targets:
                        found = n
                        break

        path = [start]
        p = found
        while p != src:
            path.append(p)
            p = parent[p]
        path.append(src)
        path.reverse()

        world = np.array(
            (self.ox + c * self.res, self.oy + (self.h - 1 - r) * self.res)
            for r, c in path
        )
        self.centerline = savgol_filter(world, smooth_window, 3, axis=0, mode="wrap")
        self.diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(self.diffs[:, 1], self.diffs[:, 0])
        self.cum_dist = np.cumsum(np.linalg.norm(self.diffs, axis=1))


def main(yaml_path: Path):
    map = Map(yaml_path)


if __name__ == "__main__":
    run(main)
