import numpy as np
import warp as wp
import warp.render
import pyglet
from imgui_bundle import imgui
from imgui_bundle.python_backends import pyglet_backend

MU = 1.0489
LF = 0.15875
LR = 0.17145
LWB = LF + LR
MASS = 3.74

STEER_MIN = -0.4189
STEER_MAX = 0.4189
STEER_V_MAX = 3.2
A_MAX = 9.51
V_MIN = -5.0
V_MAX = 20.0
PSI_PRIME_MAX = 6.0
BETA_MAX = 1.2

# Car
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

PROGRESS_SCALE = 100.0
PROGRESS_V_COEF = 10.0
TERM_PENALTY = 10.0

NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
LIDAR_RANGE = 20.0
NUM_LOOKAHEAD = 10
OBS_FRENET_OFF = 3 + NUM_LIDAR
OBS_LOOK_OFF = OBS_FRENET_OFF + 2
OBS_DIM = OBS_LOOK_OFF + 2 * NUM_LOOKAHEAD
ACT_DIM = 2
MAX_STEPS = 10_000

OCC_THRESH = 230
SMOOTH_WINDOW = 51
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DONE_TERMINATED = 1
DONE_TRUNCATED = 2

@wp.struct
class VDeriv:
    d_x: float
    d_y: float
    d_psi: float
    d_psip: float
    d_beta: float
    d_v: float

@wp.func
def st_deriv(
    delta: float,
    v: float,
    psi: float,
    psip: float,
    beta: float,
    steer_v: float,
    accel: float,
    mu_s: float,
    mass_s: float,
    lf_s: float,
    lr_s: float,
) -> VDeriv:
    lf = LF * lf_s
    lr = LR * lr_s
    lwb = lf + lr
    mu = MU * mu_s
    a_max = mu * G

    tand = wp.tan(delta)
    d_psi_kin = v * tand / lwb
    d_psi_cap = a_max / wp.max(wp.abs(v), 0.5)
    d_psi = wp.clamp(d_psi_kin, -d_psi_cap, d_psi_cap)

    a_lat = v * d_psi
    a_long_max = wp.sqrt(wp.max(a_max * a_max - a_lat * a_lat, 0.0))

    cp = wp.cos(psi)
    sp = wp.sin(psi)
    out = VDeriv()
    out.d_x = v * cp
    out.d_y = v * sp
    out.d_psi = d_psi
    out.d_v = wp.clamp(accel, -a_long_max, a_long_max)
    out.d_psip = 0.0
    out.d_beta = 0.0
    return out

@wp.func
def rk4_step(
    delta: float,
    v: float,
    psi: float,
    psip: float,
    beta: float,
    steer_v: float,
    accel: float,
    mu_s: float,
    mass_s: float,
    lf_s: float,
    lr_s: float,
) -> VDeriv:
    dd = steer_v * DT_SUB_HALF
    dd_full = steer_v * DT_SUB

    k1 = st_deriv(delta, v, psi, psip, beta, steer_v, accel, mu_s, mass_s, lf_s, lr_s)
    k2 = st_deriv(
        delta + dd,
        v + k1.d_v * DT_SUB_HALF,
        psi + k1.d_psi * DT_SUB_HALF,
        psip + k1.d_psip * DT_SUB_HALF,
        beta + k1.d_beta * DT_SUB_HALF,
        steer_v,
        accel,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k3 = st_deriv(
        delta + dd,
        v + k2.d_v * DT_SUB_HALF,
        psi + k2.d_psi * DT_SUB_HALF,
        psip + k2.d_psip * DT_SUB_HALF,
        beta + k2.d_beta * DT_SUB_HALF,
        steer_v,
        accel,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k4 = st_deriv(
        delta + dd_full,
        v + k3.d_v * DT_SUB,
        psi + k3.d_psi * DT_SUB,
        psip + k3.d_psip * DT_SUB,
        beta + k3.d_beta * DT_SUB,
        steer_v,
        accel,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    out = VDeriv()
    out.d_x = (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT_SUB_SIX
    out.d_y = (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT_SUB_SIX
    out.d_psi = (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT_SUB_SIX
    out.d_v = (k1.d_v + 2.0 * k2.d_v + 2.0 * k3.d_v + k4.d_v) * DT_SUB_SIX
    out.d_psip = (
        k1.d_psip + 2.0 * k2.d_psip + 2.0 * k3.d_psip + k4.d_psip
    ) * DT_SUB_SIX
    out.d_beta = (
        k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta
    ) * DT_SUB_SIX
    return out

@wp.kernel
def step_kernel(
    actions: wp.array(dtype=wp.vec2),
    obs: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    cars: wp.array2d(dtype=wp.float32),
    cars_int: wp.array2d(dtype=wp.int32),
    car_dr: wp.array2d(dtype=wp.float32),
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
    x = cars[i, 0]
    y = cars[i, 1]
    delta = cars[i, 2]
    v = cars[i, 3]
    psi = cars[i, 4]
    psip = cars[i, 5]
    beta = cars[i, 6]
    steps = cars_int[i, 0]
    wp_i = cars_int[i, 1]
    mu_s = car_dr[i, 0]
    mass_s = car_dr[i, 1]
    lf_s = car_dr[i, 2]
    lr_s = car_dr[i, 3]

    mw = dt_map.shape[0]
    mh = dt_map.shape[1]
    mh_f = wp.float32(mh) - 1.0

    # Input
    steer_v = wp.clamp(actions[i][0], -1.0, 1.0) * STEER_V_MAX
    if (steer_v < 0.0 and delta <= STEER_MIN) or (steer_v > 0.0 and delta >= STEER_MAX):
        steer_v = 0.0
    accel = wp.clamp(actions[i][1], -1.0, 1.0) * A_MAX
    if (accel < 0.0 and v <= V_MIN) or (accel > 0.0 and v >= V_MAX):
        accel = 0.0

    dd_sub = steer_v * DT_SUB
    for _ in range(SUBSTEPS):
        d = rk4_step(
            delta,
            v,
            psi,
            psip,
            beta,
            steer_v,
            accel,
            mu_s,
            mass_s,
            lf_s,
            lr_s,
        )
        x += d.d_x
        y += d.d_y
        delta += dd_sub
        v += d.d_v
        psi += d.d_psi
        psip += d.d_psip
        beta += d.d_beta

    delta = wp.clamp(delta, STEER_MIN, STEER_MAX)
    v = wp.clamp(v, V_MIN, V_MAX)
    psip = wp.clamp(psip, -PSI_PRIME_MAX, PSI_PRIME_MAX)
    beta = wp.clamp(beta, -BETA_MAX, BETA_MAX)

    # Reward + done
    px = wp.clamp(wp.int32((x - origin[0]) / res), 0, mw - 1)
    py = wp.clamp(wp.int32(mh_f - (y - origin[1]) / res), 0, mh - 1)
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
    v_along = v * wp.cos(beta + psi - cth)
    progress = (
        wp.float32(d_wp)
        / wp.float32(n_cl)
        * PROGRESS_SCALE
        * (1.0 + wp.max(v_along, 0.0) / PROGRESS_V_COEF)
    )

    term_pen = wp.where(term, -TERM_PENALTY, 0.0)
    reward[i] = progress + term_pen

    if term:
        done[i] = DONE_TERMINATED
    elif trunc:
        done[i] = DONE_TRUNCATED
    else:
        done[i] = 0

    # Reset
    if term or trunc:
        rng = wp.rand_init(seed_base + i * 73 + steps * 31 + new_wp * 17)
        rnd = wp.int32(wp.randf(rng) * wp.float32(n_cl)) % n_cl
        rpt = centerline[rnd]
        x = rpt[0]
        y = rpt[1]
        psi = rpt[2]
        delta = 0.0
        v = 0.0
        psip = 0.0
        beta = 0.0
        steps = 0
        new_wp = rnd
        car_dr[i, 0] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 1] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 2] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 3] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)

    # Lidar
    sh = wp.sin(psi)
    ch = wp.cos(psi)
    lx = x + LF * ch
    ly = y + LF * sh
    lpx = wp.clamp(wp.int32((lx - origin[0]) / res), 0, mw - 1)
    lpy = wp.clamp(wp.int32(mh_f - (ly - origin[1]) / res), 0, mh - 1)
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
        obs[i, 3 + j] = wp.min(dist, lrange_px) * res

    # Frenet + lookahead
    cpt = centerline[new_wp]
    cx_p = cpt[0]
    cy_p = cpt[1]
    cth_p = cpt[2]
    s_cth = wp.sin(cth_p)
    c_cth = wp.cos(cth_p)
    heading_err = wp.atan2(s_cth * ch - c_cth * sh, c_cth * ch + s_cth * sh)
    lateral_err = -(x - cx_p) * s_cth + (y - cy_p) * c_cth
    obs[i, OBS_FRENET_OFF] = heading_err
    obs[i, OBS_FRENET_OFF + 1] = lateral_err

    idx = new_wp
    for k in range(NUM_LOOKAHEAD):
        idx += look_step
        if idx >= n_cl:
            idx -= n_cl
        w = centerline[idx]
        dx = w[0] - x
        dy = w[1] - y
        obs[i, OBS_LOOK_OFF + k * 2] = dx * ch + dy * sh
        obs[i, OBS_LOOK_OFF + k * 2 + 1] = -dx * sh + dy * ch

    obs[i, 0] = delta
    obs[i, 1] = v
    obs[i, 2] = psip
    cars[i, 0] = x
    cars[i, 1] = y
    cars[i, 2] = delta
    cars[i, 3] = v
    cars[i, 4] = psi
    cars[i, 5] = psip
    cars[i, 6] = beta
    cars_int[i, 0] = steps
    cars_int[i, 1] = new_wp

from cv2 import (
    COLOR_GRAY2RGB,
    IMREAD_GRAYSCALE,
    cvtColor,
    fillPoly,
    imread,
    polylines,
)
from yaml import safe_load
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from collections import deque
from pathlib import Path

class Map:
    def __init__(self, path: Path):
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
    def _neighbors(skel, r, c, h, w):
        return [
            (r + dr, c + dc)
            for dr, dc in ADJ
            if 0 <= r + dr < h and 0 <= c + dc < w and skel[r + dr, c + dc]
        ]

    def _compute_centerline(self, free):
        skel = skeletonize(free)
        h, w = skel.shape
        pts = np.argwhere(skel)
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start = tuple(int(x) for x in pts[np.argmin(((pts - origin_px) ** 2).sum(1))])
        nbrs = self._neighbors(skel, start[0], start[1], h, w)
        if len(nbrs) < 2:
            raise RuntimeError(f"Skeleton seed {start} has {len(nbrs)} neighbours")
        src, target = nbrs[0], nbrs[1]
        parent = {src: src}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(skel, r, c, h, w):
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
        self.centerline = savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        avg_sp = float(np.linalg.norm(diffs, axis=1).mean())
        self.look_step = max(1, int(round(1.0 / avg_sp)))

    def _build_lut(self):
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        tree = KDTree(cl_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        self.lut = tree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)

class ImGuiManager:
    # ImGUI uses Top Left origin while Pyglet is Bottom Left and you have to remember DPI scaling!
    def __init__(self, renderer:warp.render.OpenGLRenderer):
        imgui.create_context()
        self.renderer = renderer

        # Tell the backend NOT to attach its broken callbacks and override them
        self.impl = pyglet_backend.create_renderer(self.renderer.window, attach_callbacks=False)
        self.impl.on_mouse_motion = self.on_mouse_motion
        self.impl.on_mouse_drag = self.on_mouse_drag
        self.impl._attach_callbacks(self.renderer.window)

        # "self.renderer.enable_keyboard_interaction = False" is not working dynamically
        self.renderer.window.push_handlers(on_key_press=self.on_key_press)

    def want_capture_mouse(self, *args, **kwargs):
        return self.impl.io.want_capture_mouse

    def on_mouse_motion(self, x, y, dx, dy):
        ratio = self.renderer.window.get_pixel_ratio()
        self.impl.io.add_mouse_pos_event(x / ratio, self.impl.io.display_size.y - (y / ratio))
    
    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        self.impl._on_mouse_button(button, True)
        self.on_mouse_motion(x, y, dx, dy)
        return self.impl.io.want_capture_mouse
    
    def on_key_press(self, symbol, modifiers):
        return self.impl.io.want_capture_keyboard

    def render_frame(self):
        """Renders a single frame of the UI. This should be called from the main render loop."""
        io = imgui.get_io()
        ratio = self.renderer.window.get_pixel_ratio()
        io.display_size = self.renderer.screen_width / ratio, self.renderer.screen_height / ratio

        self.impl.process_inputs()
        imgui.new_frame()

        self.draw_ui()

        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def draw_ui(self):
        """Draws the UI"""
        imgui.set_next_window_size(imgui.ImVec2(640, 480), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos(imgui.ImVec2(0, 0), imgui.Cond_.first_use_ever)

        imgui.begin("Warp Float Values")

        imgui.text(f"A read-only float: {self.renderer.clock_time}")
        imgui.separator()

        imgui.text("Editable floats:")
        imgui.separator()
        imgui.text("File Dialog Examples:")

        static_text = "Hello, World!"
        imgui.input_text("What does the fox say?", static_text)

        imgui.end()

    def shutdown(self):
        self.impl.shutdown()

class Visuals:
    # If you see this error and you have more than one GPU (iGPU & eGPU):
    #   "Warp UserWarning: Could not register GL buffer since CUDA/OpenGL interoperability is not available.
    #   Falling back to copy operations between the Warp array and the OpenGL buffer."
    # Then you have to make sure ALL aspects of the Python program is running on GPU. On Windows you find
    # the Python executable and set to "High Performance" in Windows Graphics settings.

    def __init__(self):
        print(f"Warp Device: {wp.get_device().name}")
        print(f"Pyglet Device: {pyglet.gl.gl_info.get_renderer()}")
    
        # Init Warp Render
        self.renderer = warp.render.OpenGLRenderer(
            fps=60,
            screen_width=1280,
            screen_height=720,
            up_axis="Z", # Cannot change sun direction because it's hard coded in! Cringe!
            device=wp.get_device()
        )

        # Init ImGUI
        self.imgui_manager = ImGuiManager(self.renderer)
        self.renderer.render_2d_callbacks.append(self.imgui_manager.render_frame)

        # Setup rendering loop
        pyglet.clock.schedule_interval(self.pyglet_draw, self.renderer._frame_dt)
        pyglet.app.run()
        self.clear()

    def pyglet_draw(self, dt):
        self.render()

    def render(self):
        # Warp Render Begin
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)

        # Begin Render
        # Example shapes
        self.renderer.render_cylinder(
            "cylinder",
            [3.2, 1.0, np.sin(time + 0.5)],
            np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
            radius=0.5,
            half_height=0.8,
        )
        self.renderer.render_cone(
            "cone",
            [-1.2, 1.0, 0.0],
            np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
            radius=0.5,
            half_height=0.8,
        )
        # End Render

        # Warp Render End
        self.renderer.end_frame()

    def clear(self):
        self.imgui_manager.shutdown()
        self.renderer.clear()

if __name__ == "__main__":
    with wp.ScopedDevice("cuda"):
        vs = Visuals()