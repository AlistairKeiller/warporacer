# pyright: reportIndexIssue=false
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
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
            * wp.cos(car_delta) ** 2.0
            * (1.0 + (wp.tan(car_delta) ** 2.0 * LENGTH_REAR / LENGTH_WHEELBASE) ** 2.0)
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
        state.dd_psi = (
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
    car_px = wp.int32((car_x - origin_x) / res)
    car_py = wp.int32(
        float(distance_transform_px.shape[1]) - 1.0 - (car_y - origin_y) / res
    )

    car_pos_px = wp.vec2(wp.float32(car_px), wp.float32(car_py))

    # collision logic
    term = distance_transform_px[car_px, car_py] * res < wp.length(
        wp.vec2(WIDTH / 2.0, LENGTH / 2.0)
    )
    trunc = car_steps >= MAX_STEPS
    car_steps += 1

    # reward logic
    new_car_waypoint = centerline_lut[car_px, car_py]
    d_centerline_pt = new_car_waypoint - car_waypoint
    if d_centerline_pt > num_centerline_pts / 2:
        d_centerline_pt -= num_centerline_pts
    elif d_centerline_pt < -num_centerline_pts / 2:
        d_centerline_pt += num_centerline_pts
    reward[i] = wp.float32(d_centerline_pt) / wp.float32(
        num_centerline_pts
    ) - wp.float32(term)

    # reset logic
    if trunc or term:
        random_number = (
            wp.int32(wp.uint32(i * 2654435761) >> wp.uint32(16)) % num_centerline_pts
        )
        car_x = centerline[random_number][0]
        car_y = centerline[random_number][1]
        car_delta = 0.0
        car_v = 0.0
        car_psi = centerline[random_number][2]
        car_psi_prime = 0.0
        car_beta = 0.0
        car_steps = 0
        new_car_waypoint = random_number

    # raycast
    sh, ch = wp.sin(car_psi), wp.cos(car_psi)
    for j in range(lidar_dirs.shape[0]):
        ray = wp.vec2(car_px, car_py)
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        d_px = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        while wp.length(ray - car_pos_px) * res < LIDAR_RANGE:
            ray_px = wp.int32(ray[0])
            ray_py = wp.int32(ray[1])
            dt_ray = distance_transform_px[ray_px, ray_py]
            ray += d_px * dt_ray
            if dt_ray == 0.0:
                break
        observation[i, j + 3] = wp.length(ray - car_pos_px) * res

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
        # self.occupied = self.raw < OCC_THRESH
        self.dt = distance_transform_edt(self.raw >= OCC_THRESH)
        self.ox, self.oy, self.ophi = self.meta["origin"]
        # self.h, self.w = self.occupied.shape
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
        # self.cum_dist = np.cumsum(np.linalg.norm(self.diffs, axis=1))

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


def main(yaml_path: Path):
    map = Map(yaml_path)

    try:
        import rerun as rr

        rr.init("warporacer")
        rr.spawn()

        rr.log("image/raw", rr.Image(map.raw))
        rr.log("image/world", rr.Image(map.raw))
        rr.log("image/dt", rr.Image(distance_transform_edt(map.raw >= OCC_THRESH)))
        centerline_px = np.column_stack(
            [
                (map.centerline[:, 0] - map.ox) / map.res,
                map.h - 1 - (map.centerline[:, 1] - map.oy) / map.res,
            ]
        )
        rr.log("image/world/centerline", rr.Points2D(centerline_px, radii=1.0))
    except ImportError:
        pass


if __name__ == "__main__":
    run(main)
