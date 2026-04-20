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
    car_px = wp.clamp(wp.int32((car_x - origin_x) / res), 0, map_w - 1)
    car_py = wp.clamp(
        wp.int32(float(map_h) - 1.0 - (car_y - origin_y) / res), 0, map_h - 1
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

    # raycast
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


def main(
    yaml_path: Path,
    num_cars: int = 4,
    num_steps: int = 2000,
    substeps: int = 4,
    seed: int = 0,
):
    """
    Debugging driver. Spawns `num_cars` cars at random centerline points, drives
    them with a gentle throttle + noisy steer, and streams everything to rerun.

    Logged per step:
      map/car_k/pos       — car position
      map/car_k/heading   — short white line showing psi
      map/car_k/lidar     — 108 lidar rays
      telem/car_k/{v, delta, psi_dot, beta, reward, steps}
    """
    import rerun as rr

    wp.init()
    rr.init("warporacer")
    rr.spawn()

    rng = np.random.default_rng(seed)
    m = Map(yaml_path)

    # static layers
    rr.log("map", rr.Image(m.raw), static=True)
    rr.log(
        "map/dt",
        rr.Image((m.dt / max(m.dt.max(), 1.0) * 255).astype(np.uint8)),
        static=True,
    )
    centerline_px_disp = np.column_stack(
        [
            (m.centerline[:, 0] - m.ox) / m.res,
            m.h - 1 - (m.centerline[:, 1] - m.oy) / m.res,
        ]
    )
    rr.log(
        "map/centerline",
        rr.Points2D(centerline_px_disp, colors=[(0, 120, 255)], radii=0.4),
        static=True,
    )

    # lidar beams: 108 rays over ±135°, matches observation layout
    num_lidar = 108
    lidar_angles = np.linspace(-np.radians(135), np.radians(135), num_lidar).astype(
        np.float32
    )
    lidar_dirs_np = np.column_stack(
        [np.cos(lidar_angles), np.sin(lidar_angles)]
    ).astype(np.float32)

    # warp map buffers (transposed because kernel indexes [x, y] not [row, col])
    dt_wp = wp.array(m.dt.T.astype(np.float32), dtype=float)
    lut_wp = wp.array(m.centerline_lut.T.astype(np.int32), dtype=int)
    cl_wp = wp.array(
        np.column_stack([m.centerline, m.angles]).astype(np.float32), dtype=wp.vec3
    )
    lidar_wp = wp.array(lidar_dirs_np, dtype=wp.vec2)
    n_cl = len(m.centerline)

    # spawn cars at random centerline points
    spawn_idxs = rng.integers(0, n_cl, size=num_cars)
    cars_np = np.zeros((num_cars, 7), dtype=np.float32)
    cars_int_np = np.zeros((num_cars, 2), dtype=np.int32)
    for k, idx in enumerate(spawn_idxs):
        cars_np[k] = [
            m.centerline[idx, 0],
            m.centerline[idx, 1],
            0.0,
            0.0,
            m.angles[idx],
            0.0,
            0.0,
        ]
        cars_int_np[k, 0] = 0
        cars_int_np[k, 1] = int(idx)

    cars_wp = wp.array(cars_np, dtype=float)
    cars_int_wp = wp.array(cars_int_np, dtype=int)
    obs_wp = wp.zeros((num_cars, 3 + num_lidar), dtype=float)
    rew_wp = wp.zeros(num_cars, dtype=float)

    # sanity: warn if any spawn starts inside the collision skin
    collision_radius = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0))
    for k, idx in enumerate(spawn_idxs):
        wx, wy = m.centerline[idx, 0], m.centerline[idx, 1]
        ix = int(np.clip((wx - m.ox) / m.res, 0, m.w - 1))
        iy = int(np.clip(m.h - 1 - (wy - m.oy) / m.res, 0, m.h - 1))
        clearance = m.dt[iy, ix] * m.res
        if clearance < collision_radius:
            print(
                f"WARN car_{k} spawn idx={idx} clearance={clearance:.3f}m "
                f"< collision_radius={collision_radius:.3f}m — will reset immediately"
            )

    car_colors = [
        (255, 64, 64),
        (64, 255, 255),
        (255, 255, 64),
        (255, 128, 255),
        (128, 255, 128),
        (255, 160, 64),
    ]

    for t in range(num_steps):
        # gentle wandering driver
        act_np = np.zeros((num_cars, 2), dtype=np.float32)
        act_np[:, 0] = rng.uniform(-0.25, 0.25, size=num_cars)
        act_np[:, 1] = 0.4
        act_wp = wp.array(act_np, dtype=wp.vec2)

        for _ in range(substeps):
            wp.launch(
                step,
                dim=num_cars,
                inputs=[
                    act_wp,
                    obs_wp,
                    rew_wp,
                    cars_wp,
                    cars_int_wp,
                    wp.vec2(m.ox, m.oy),
                    m.res,
                    dt_wp,
                    lut_wp,
                    cl_wp,
                    n_cl,
                    lidar_wp,
                ],
            )
        wp.synchronize()

        cars_out = cars_wp.numpy()
        obs_out = obs_wp.numpy()
        rew_out = rew_wp.numpy()
        steps_out = cars_int_wp.numpy()[:, 0]

        rr.set_time("step", sequence=t)

        for k in range(num_cars):
            c = cars_out[k]
            o = obs_out[k]
            px = (c[0] - m.ox) / m.res
            py = m.h - 1 - (c[1] - m.oy) / m.res
            psi = c[4]
            sh, ch = np.sin(psi), np.cos(psi)

            # lidar ends: obs is in meters, convert to pixels for overlay
            dists_m = o[3 : 3 + num_lidar]
            dists_px = dists_m / m.res
            dx = ch * lidar_dirs_np[:, 0] - sh * lidar_dirs_np[:, 1]
            dy = -(sh * lidar_dirs_np[:, 0] + ch * lidar_dirs_np[:, 1])
            ends = np.column_stack([px + dx * dists_px, py + dy * dists_px])
            starts = np.broadcast_to([px, py], ends.shape)
            strips = np.stack([starts, ends], axis=1)

            color = car_colors[k % len(car_colors)]
            rr.log(
                f"map/car_{k}/pos",
                rr.Points2D([[px, py]], radii=4.0, colors=[color]),
            )
            rr.log(
                f"map/car_{k}/heading",
                rr.LineStrips2D(
                    [[[px, py], [px + ch * 12.0, py - sh * 12.0]]],
                    colors=[(255, 255, 255)],
                ),
            )
            rr.log(
                f"map/car_{k}/lidar",
                rr.LineStrips2D(strips, colors=[color + (70,)]),
            )

            rr.log(f"telem/car_{k}/v", rr.Scalars(float(c[3])))
            rr.log(f"telem/car_{k}/delta", rr.Scalars(float(c[2])))
            rr.log(f"telem/car_{k}/psi_dot", rr.Scalars(float(c[5])))
            rr.log(f"telem/car_{k}/beta", rr.Scalars(float(c[6])))
            rr.log(f"telem/car_{k}/reward", rr.Scalars(float(rew_out[k])))
            rr.log(f"telem/car_{k}/steps", rr.Scalars(float(steps_out[k])))


if __name__ == "__main__":
    run(main)
