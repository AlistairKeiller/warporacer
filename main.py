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
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@wp.kernel
def scan_lidars(
    cars: wp.array[wp.vec3],
    origin: wp.vec2,
    res: float,
    dt: wp.array2d[float],
    lidar_dirs: wp.array[wp.vec2],
    max_distance: float,
    lidar_ranges: wp.array2d[float],
):
    i = wp.tid()
    car = cars[i]
    car_px = (car[0] - origin[0]) / res
    car_py = dt.size[1] - 1 - (car[1] - origin[1]) / res
    car_phi = car[2]
    ray = wp.vec2(car_px, car_py)
    sh, ch = wp.sin(car_phi), wp.cos(car_phi)
    for j in range(len(lidar_dirs)):
        ca, sa = lidar_dirs[j]
        d_px = wp.vec2(ch * ca - sh * sa, sh * ca + ch * sa)
        while wp.length(ray - wp.vec2(car_px, car_py)) < max_distance:
            ray_px = wp.int32(ray[0])
            ray_py = wp.int32(ray[1])
            dt_ray = dt[ray_px, ray_py]
            ray += d_px * dt_ray
            if dt_ray == 0.0:
                lidar_ranges[i, j] = wp.length(ray - wp.vec2(car_px, car_py))
                break


class Map:
    def __init__(self, path: Path, lidar_max_dist: float) -> None:
        with open(path, "r") as f:
            self.meta = safe_load(f)
        raw = imread(str(path.parent / self.meta["image"]), IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(path.parent / self.meta["image"])
        self.occupied = wp.array(raw < OCC_THRESH)
        self.dt = wp.array(distance_transform_edt(raw >= OCC_THRESH))
        self.lidar_range = lidar_max_dist
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


def main(yaml_path: Path, lidar_range: float = 20.0):
    map = Map(yaml_path, lidar_range)


if __name__ == "__main__":
    run(main)
