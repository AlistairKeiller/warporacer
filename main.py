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

# Masses
MASS = 3.74
MASS_S = 3.34  # sprung
MASS_UF = 0.20  # unsprung front
MASS_UR = 0.20  # unsprung rear

# Geometry
A = 0.15875  # CG -> front axle
B = 0.17145  # CG -> rear axle
T_F = 0.29  # track width front
T_R = 0.29  # track width rear
H_S = 0.074  # sprung CG height
H_RAF = 0.0  # front roll axis height
H_RAR = 0.0  # rear roll axis height
R_W = 0.0505  # wheel radius

# Inertias
I_Z = 0.04712
I_PHI_S = 0.0325  # sprung roll
I_Y_S = 0.0781  # sprung pitch
I_XZ_S = 0.0
I_UF = 0.00432  # unsprung roll, front axle (~m_uf * (T_F/2)^2)
I_UR = 0.00432
I_Y_W = 2.0e-4  # wheel rotational (cylinder model + drivetrain reflected)

# Suspension
K_SF = 1000.0  # spring N/m (per corner)
K_SR = 1000.0
K_SDF = 15.0  # damper N*s/m
K_SDR = 15.0
K_ZT = 10000.0  # vertical tire stiffness
K_RAS = 3000.0  # compliant-pin lateral stiffness
K_RAD = 30.0  # compliant-pin lateral damping
K_TSF = -25.0  # aux torsion roll stiffness (front)
K_TSR = 0.0
K_LT = 4.0e-4  # tire lateral compliance
D_F = -4.5  # camber vs stroke (front)
D_R = -1.5
E_F = 0.0
E_R = 0.0

# Drivetrain split (T_SE=0 -> RWD; T_SB=0.6 -> 60% brakes to front)
T_SB = 0.6
T_SE = 0.0

# Input limits
STEER_MIN, STEER_MAX = -0.4189, 0.4189
STEER_V_MAX = 3.2
A_MAX = 9.51
V_MIN, V_MAX = -5.0, 20.0
V_SWITCH = 7.319

# Pacejka Magic Formula (from parameters_vehicle1.yaml)
P_CX1, P_DX1, P_DX3, P_EX1 = 1.6411, 1.1739, 0.0, 0.46403
P_KX1, P_HX1, P_VX1 = 22.303, 0.0012297, -8.8098e-6
R_BX1, R_BX2, R_CX1, R_EX1, R_HX1 = 13.276, -13.778, 1.2568, 0.65225, 0.0050722
P_CY1, P_DY1, P_DY3, P_EY1 = 1.3507, 1.0489, -2.8821, -0.0074722
P_KY1, P_HY1, P_HY3 = -21.92, 0.0026747, 0.031415
P_VY1, P_VY3 = 0.037318, -0.32931
R_BY1, R_BY2, R_BY3 = 7.1433, 9.1916, -0.027856
R_CY1, R_EY1, R_HY1 = 1.0719, -0.27572, 5.7448e-6
R_VY1, R_VY3, R_VY4 = -0.027825, -0.27568, 12.12
R_VY5, R_VY6 = 1.9, -10.704

# Slip regularisation -- replaces CommonRoad's |v|<0.1 kinematic fallback
SLIP_EPS = 0.5  # m/s: floor on |u_w| in longitudinal slip denom
ATAN_EPS = 0.01  # floor on denom inside atan for slip angle

# Static-equilibrium suspension compression so a reset car isn't in free-fall.
# Solve coupled 3x3 for (zs, zuf, zur):
#   sprung:  (K_SF + K_SR)*zs - K_SF*zuf - K_SR*zur = 0
#   unsprung F:  -K_SF*zs + (K_ZT + K_SF/2)*2*zuf .. (after reduction)
# For the symmetric case K_SF=K_SR, zs = (zuf + zur)/2 and
#   (K_ZT + K_SF/2)*zu - (K_SF/2)*zu' = w_corner + 0.5*m_u*g
_W_F = 3.34 * 9.81 * 0.17145 / (2.0 * (0.15875 + 0.17145))
_W_R = 3.34 * 9.81 * 0.15875 / (2.0 * (0.15875 + 0.17145))
_den = (K_ZT + K_SF / 2.0) ** 2 - (K_SF / 2.0) ** 2
ZUF_STATIC = (
    (K_ZT + K_SF / 2.0) * (_W_F + 0.5 * MASS_UF * 9.81)
    + (K_SF / 2.0) * (_W_R + 0.5 * MASS_UR * 9.81)
) / _den
ZUR_STATIC = (
    (K_ZT + K_SR / 2.0) * (_W_R + 0.5 * MASS_UR * 9.81)
    + (K_SR / 2.0) * (_W_F + 0.5 * MASS_UF * 9.81)
) / _den
ZS_STATIC = (ZUF_STATIC + ZUR_STATIC) / 2.0

# Sim steps
DT = 1.0 / 60.0
SUBSTEPS = 6
DT_SUB = DT / float(SUBSTEPS)
DT_SUB_HALF = DT_SUB * 0.5
DT_SUB_SIX = DT_SUB / 6.0
G = 9.81

# car dimensions
WIDTH, LENGTH = 0.31, 0.58
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0))

# LIDAR parameters
NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
LIDAR_RANGE = 20.0
NUM_LOOKAHEAD = 10

# observation dimensions
OBS_PROP = 4  # delta, u, v, psip
OBS_LIDAR_OFF = OBS_PROP
OBS_FRENET_OFF = OBS_PROP + NUM_LIDAR
OBS_LOOK_OFF = OBS_FRENET_OFF + 2
OBS_DIM = OBS_LOOK_OFF + 2 * NUM_LOOKAHEAD
ACT_DIM = 2

# simulation parameters
MAX_STEPS = 10_000
PROGRESS_SCALE = 10.0
WALL_PENALTY_COEF = 0.05
WALL_PENALTY_RATE = 3.0
TERM_PENALTY = 20.0

# occupancy grid parameters
OCC_THRESH = 230
SMOOTH_WINDOW = 51
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DONE_TERMINATED = 1
DONE_TRUNCATED = 2

# State vector (matches CommonRoad MB x(1..29) in order, 0-indexed):
#   0:sx   1:sy   2:delta 3:u    4:psi   5:psip
#   6:phi  7:phip 8:theta 9:thetap 10:v  11:zs  12:ws
#   13:phif 14:phifp 15:vuf 16:zuf 17:wuf
#   18:phir 19:phirp 20:vur 21:zur 22:wur
#   23:wlf 24:wrf 25:wlr 26:wrr  27:dyf 28:dyr
vec29 = wp.types.vector(length=29, dtype=wp.float32)


@wp.func
def reset29(sx: float, sy: float, psi: float) -> vec29:
    return vec29(
        sx,
        sy,
        0.0,
        3.0,
        psi,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        ZS_STATIC,
        0.0,
        0.0,
        0.0,
        0.0,
        ZUF_STATIC,
        0.0,
        0.0,
        0.0,
        0.0,
        ZUR_STATIC,
        0.0,
        59.4,
        59.4,
        59.4,
        59.4,
        0.0,
        0.0,
    )


@wp.func
def sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@wp.func
def clamp_away(x: float, eps: float) -> float:
    if x >= 0.0:
        return wp.max(x, eps)
    return wp.min(x, -eps)


@wp.func
def pacejka_long(kappa: float, gamma: float, fz: float) -> float:
    k = -kappa
    kx = k + P_HX1
    mu_x = P_DX1 * (1.0 - P_DX3 * gamma * gamma)
    dx = mu_x * fz
    bx = (fz * P_KX1) / (P_CX1 * dx + 1e-6)
    bk = bx * kx
    return dx * wp.sin(P_CX1 * wp.atan(bk - P_EX1 * (bk - wp.atan(bk)))) + fz * P_VX1


@wp.func
def pacejka_lat(alpha: float, gamma: float, fz: float) -> wp.vec2:
    sg = sign(gamma)
    ag = wp.abs(gamma)
    shy = sg * (P_HY1 + P_HY3 * ag)
    svy = sg * fz * (P_VY1 + P_VY3 * ag)
    ay = alpha + shy
    mu_y = P_DY1 * (1.0 - P_DY3 * gamma * gamma)
    dy = mu_y * fz
    by = (fz * P_KY1) / (P_CY1 * dy + 1e-6)
    bk = by * ay
    fy = dy * wp.sin(P_CY1 * wp.atan(bk - P_EY1 * (bk - wp.atan(bk)))) + svy
    return wp.vec2(fy, mu_y)


@wp.func
def pacejka_long_comb(kappa: float, alpha: float, fx0: float) -> float:
    bxa = R_BX1 * wp.cos(wp.atan(R_BX2 * kappa))
    ref = bxa * R_HX1 - R_EX1 * (bxa * R_HX1 - wp.atan(bxa * R_HX1))
    dxa = fx0 / (wp.cos(R_CX1 * wp.atan(ref)) + 1e-6)
    inner = bxa * (alpha + R_HX1) - R_EX1 * (
        bxa * (alpha + R_HX1) - wp.atan(bxa * (alpha + R_HX1))
    )
    return dxa * wp.cos(R_CX1 * wp.atan(inner))


@wp.func
def pacejka_lat_comb(
    kappa: float,
    alpha: float,
    gamma: float,
    mu_y: float,
    fz: float,
    fy0: float,
) -> float:
    bk = R_BY1 * wp.cos(wp.atan(R_BY2 * (alpha - R_BY3)))
    ref = bk * R_HY1 - R_EY1 * (bk * R_HY1 - wp.atan(bk * R_HY1))
    dyk = fy0 / (wp.cos(R_CY1 * wp.atan(ref)) + 1e-6)
    dvyk = mu_y * fz * (R_VY1 + R_VY3 * gamma) * wp.cos(wp.atan(R_VY4 * alpha))
    svyk = dvyk * wp.sin(R_VY5 * wp.atan(R_VY6 * kappa))
    ks = kappa + R_HY1
    inner = bk * ks - R_EY1 * (bk * ks - wp.atan(bk * ks))
    return dyk * wp.cos(R_CY1 * wp.atan(inner)) + svyk


@wp.func
def mb_deriv(s: vec29, vdelta: float, accel: float) -> vec29:
    # Unpack state
    delta = s[2]
    u = s[3]
    psi = s[4]
    psip = s[5]
    phi = s[6]
    phip = s[7]
    theta = s[8]
    thetap = s[9]
    v = s[10]
    zs = s[11]
    ws = s[12]
    phif = s[13]
    phifp = s[14]
    vuf = s[15]
    zuf = s[16]
    wuf = s[17]
    phir = s[18]
    phirp = s[19]
    vur = s[20]
    zur = s[21]
    wur = s[22]
    wlf = s[23]
    wrf = s[24]
    wlr = s[25]
    wrr = s[26]
    dyf = s[27]
    dyr = s[28]

    cd = wp.cos(delta)
    sd = wp.sin(delta)
    cph = wp.cos(phi)
    sph = wp.sin(phi)
    inv_cph = 1.0 / wp.max(cph, 1e-6)  # phi stays small, safe

    # Vertical tire loads (K_zt * compression, clamped >= 0)
    cphif = wp.cos(phif)
    sphif = wp.sin(phif)
    cphir = wp.cos(phir)
    sphir = wp.sin(phir)
    fz_lf = wp.max((zuf + R_W * (cphif - 1.0) - 0.5 * T_F * sphif) * K_ZT, 0.0)
    fz_rf = wp.max((zuf + R_W * (cphif - 1.0) + 0.5 * T_F * sphif) * K_ZT, 0.0)
    fz_lr = wp.max((zur + R_W * (cphir - 1.0) - 0.5 * T_R * sphir) * K_ZT, 0.0)
    fz_rr = wp.max((zur + R_W * (cphir - 1.0) + 0.5 * T_R * sphir) * K_ZT, 0.0)

    # Contact-patch longitudinal speeds
    u_lf = (u + 0.5 * T_F * psip) * cd + (v + A * psip) * sd
    u_rf = (u - 0.5 * T_F * psip) * cd + (v + A * psip) * sd
    u_lr = u + 0.5 * T_R * psip
    u_rr = u - 0.5 * T_R * psip

    # Longitudinal slip (no kinematic fallback)
    s_lf = (u_lf - R_W * wlf) / wp.max(wp.abs(u_lf), SLIP_EPS)
    s_rf = (u_rf - R_W * wrf) / wp.max(wp.abs(u_rf), SLIP_EPS)
    s_lr = (u_lr - R_W * wlr) / wp.max(wp.abs(u_lr), SLIP_EPS)
    s_rr = (u_rr - R_W * wrr) / wp.max(wp.abs(u_rr), SLIP_EPS)

    # Lateral slip (atan with clamped denominator)
    num_f = v + A * psip - phifp * (R_W - zuf)
    num_r = v - B * psip - phirp * (R_W - zur)
    alpha_lf = wp.atan(num_f / clamp_away(u_lf, ATAN_EPS)) - delta
    alpha_rf = wp.atan(num_f / clamp_away(u_rf, ATAN_EPS)) - delta
    alpha_lr = wp.atan(num_r / clamp_away(u_lr, ATAN_EPS))
    alpha_rr = wp.atan(num_r / clamp_away(u_rr, ATAN_EPS))

    # Suspension strokes + rates
    dz_f_base = (H_S - R_W + zuf - zs) * inv_cph - H_S + R_W
    dz_r_base = (H_S - R_W + zur - zs) * inv_cph - H_S + R_W
    z_slf = dz_f_base + A * theta + 0.5 * (phi - phif) * T_F
    z_srf = dz_f_base + A * theta - 0.5 * (phi - phif) * T_F
    z_slr = dz_r_base - B * theta + 0.5 * (phi - phir) * T_R
    z_srr = dz_r_base - B * theta - 0.5 * (phi - phir) * T_R
    dz_slf = wuf - ws + A * thetap + 0.5 * (phip - phifp) * T_F
    dz_srf = wuf - ws + A * thetap - 0.5 * (phip - phifp) * T_F
    dz_slr = wur - ws - B * thetap + 0.5 * (phip - phirp) * T_R
    dz_srr = wur - ws - B * thetap - 0.5 * (phip - phirp) * T_R

    # Camber (linear with stroke)
    g_lf = phi + D_F * z_slf + E_F * z_slf * z_slf
    g_rf = phi - D_F * z_srf - E_F * z_srf * z_srf
    g_lr = phi + D_R * z_slr + E_R * z_slr * z_slr
    g_rr = phi - D_R * z_srr - E_R * z_srr * z_srr

    # Pacejka: pure + combined slip
    f0x_lf = pacejka_long(s_lf, g_lf, fz_lf)
    f0x_rf = pacejka_long(s_rf, g_rf, fz_rf)
    f0x_lr = pacejka_long(s_lr, g_lr, fz_lr)
    f0x_rr = pacejka_long(s_rr, g_rr, fz_rr)
    lat_lf = pacejka_lat(alpha_lf, g_lf, fz_lf)
    lat_rf = pacejka_lat(alpha_rf, g_rf, fz_rf)
    lat_lr = pacejka_lat(alpha_lr, g_lr, fz_lr)
    lat_rr = pacejka_lat(alpha_rr, g_rr, fz_rr)
    fx_lf = pacejka_long_comb(s_lf, alpha_lf, f0x_lf)
    fx_rf = pacejka_long_comb(s_rf, alpha_rf, f0x_rf)
    fx_lr = pacejka_long_comb(s_lr, alpha_lr, f0x_lr)
    fx_rr = pacejka_long_comb(s_rr, alpha_rr, f0x_rr)
    fy_lf = pacejka_lat_comb(s_lf, alpha_lf, g_lf, lat_lf[1], fz_lf, lat_lf[0])
    fy_rf = pacejka_lat_comb(s_rf, alpha_rf, g_rf, lat_rf[1], fz_rf, lat_rf[0])
    fy_lr = pacejka_lat_comb(s_lr, alpha_lr, g_lr, lat_lr[1], fz_lr, lat_lr[0])
    fy_rr = pacejka_lat_comb(s_rr, alpha_rr, g_rr, lat_rr[1], fz_rr, lat_rr[0])

    # Compliant pin joint forces
    dzf = H_S - R_W + zuf - zs
    dzr = H_S - R_W + zur - zs
    dpf = phi - phif
    dpr = phi - phir
    ddpf = phip - phifp
    ddpr = phip - phirp
    df = dzf * sph - dyf * cph - (H_RAF - R_W) * wp.sin(dpf)
    dr_ = dzr * sph - dyr * cph - (H_RAR - R_W) * wp.sin(dpr)
    ddf = (
        (dzf * cph + dyf * sph) * phip
        + (wuf - ws) * sph
        - (v + A * psip - vuf) * cph
        - (H_RAF - R_W) * wp.cos(dpf) * ddpf
    )
    ddr = (
        (dzr * cph + dyr * sph) * phip
        + (wur - ws) * sph
        - (v - B * psip - vur) * cph
        - (H_RAR - R_W) * wp.cos(dpr) * ddpr
    )
    f_raf = df * K_RAS + ddf * K_RAD
    f_rar = dr_ * K_RAS + ddr * K_RAD

    # Suspension vertical forces
    w_f = MASS_S * G * B / (2.0 * (A + B))
    w_r = MASS_S * G * A / (2.0 * (A + B))
    f_slf = w_f - z_slf * K_SF - dz_slf * K_SDF + (phi - phif) * K_TSF / T_F
    f_srf = w_f - z_srf * K_SF - dz_srf * K_SDF - (phi - phif) * K_TSF / T_F
    f_slr = w_r - z_slr * K_SR - dz_slr * K_SDR + (phi - phir) * K_TSR / T_R
    f_srr = w_r - z_srr * K_SR - dz_srr * K_SDR - (phi - phir) * K_TSR / T_R

    # Sprung-mass totals
    sum_x = fx_lr + fx_rr + (fx_lf + fx_rf) * cd - (fy_lf + fy_rf) * sd
    sum_n = (
        (fy_lf + fy_rf) * A * cd
        + (fx_lf + fx_rf) * A * sd
        + (fy_rf - fy_lf) * 0.5 * T_F * sd
        + (fx_lf - fx_rf) * 0.5 * T_F * cd
        + (fx_lr - fx_rr) * 0.5 * T_R
        - (fy_lr + fy_rr) * B
    )
    sum_ys = (f_raf + f_rar) * cph + (f_slf + f_slr + f_srf + f_srr) * sph
    sum_l = (
        0.5 * (f_slf - f_srf) * T_F
        + 0.5 * (f_slr - f_srr) * T_R
        - f_raf * inv_cph * (H_S - zs - R_W + zuf - (H_RAF - R_W) * cphif)
        - f_rar * inv_cph * (H_S - zs - R_W + zur - (H_RAR - R_W) * cphir)
    )
    sum_zs = (f_slf + f_slr + f_srf + f_srr) * cph - (f_raf + f_rar) * sph
    sum_ms = (
        A * (f_slf + f_srf)
        - B * (f_slr + f_srr)
        + ((fx_lf + fx_rf) * cd - (fy_lf + fy_rf) * sd + fx_lr + fx_rr) * (H_S - zs)
    )

    # Unsprung totals
    sum_luf = (
        0.5 * (f_srf - f_slf) * T_F
        - f_raf * (H_RAF - R_W)
        + fz_lf * (R_W * sphif + 0.5 * T_F * cphif - K_LT * fy_lf)
        - fz_rf * (-R_W * sphif + 0.5 * T_F * cphif + K_LT * fy_rf)
        - ((fy_lf + fy_rf) * cd + (fx_lf + fx_rf) * sd) * (R_W - zuf)
    )
    sum_lur = (
        0.5 * (f_srr - f_slr) * T_R
        - f_rar * (H_RAR - R_W)
        + fz_lr * (R_W * sphir + 0.5 * T_R * cphir - K_LT * fy_lr)
        - fz_rr * (-R_W * sphir + 0.5 * T_R * cphir + K_LT * fy_rr)
        - (fy_lr + fy_rr) * (R_W - zur)
    )
    sum_zuf = fz_lf + fz_rf + f_raf * sph - (f_slf + f_srf) * cph
    sum_zur = fz_lr + fz_rr + f_rar * sph - (f_slr + f_srr) * cph
    sum_yuf = (
        (fy_lf + fy_rf) * cd
        + (fx_lf + fx_rf) * sd
        - f_raf * cph
        - (f_slf + f_srf) * sph
    )
    sum_yur = (fy_lr + fy_rr) - f_rar * cph - (f_slr + f_srr) * sph

    # Drive/brake torques
    t_b = wp.where(accel > 0.0, 0.0, MASS * R_W * accel)
    t_e = wp.where(accel > 0.0, MASS * R_W * accel, 0.0)

    # Body velocity for world-frame position integration
    vel = wp.sqrt(u * u + v * v)
    beta = wp.atan2(v, u)

    # Coupled yaw/roll (2x2 inversion with I_xz_s cross term)
    yaw_den = I_Z - I_XZ_S * I_XZ_S / I_PHI_S
    roll_den = I_PHI_S - I_XZ_S * I_XZ_S / I_Z
    dpsip = (sum_n + I_XZ_S / I_PHI_S * sum_l) / yaw_den
    dphip = (I_XZ_S / I_Z * sum_n + sum_l) / roll_den

    return vec29(
        wp.cos(beta + psi) * vel,  # 0  d(sx)
        wp.sin(beta + psi) * vel,  # 1  d(sy)
        vdelta,  # 2  d(delta)
        sum_x / MASS + psip * v,  # 3  d(u)
        psip,  # 4  d(psi)
        dpsip,  # 5  d(psip)
        phip,  # 6  d(phi)
        dphip,  # 7  d(phip)
        thetap,  # 8  d(theta)
        sum_ms / I_Y_S,  # 9  d(thetap)
        sum_ys / MASS_S - psip * u,  # 10 d(v)
        ws,  # 11 d(zs)
        G - sum_zs / MASS_S,  # 12 d(ws)
        phifp,  # 13 d(phif)
        sum_luf / I_UF,  # 14 d(phifp)
        sum_yuf / MASS_UF - psip * u,  # 15 d(vuf)
        wuf,  # 16 d(zuf)
        G - sum_zuf / MASS_UF,  # 17 d(wuf)
        phirp,  # 18 d(phir)
        sum_lur / I_UR,  # 19 d(phirp)
        sum_yur / MASS_UR - psip * u,  # 20 d(vur)
        wur,  # 21 d(zur)
        G - sum_zur / MASS_UR,  # 22 d(wur)
        (-R_W * fx_lf + 0.5 * T_SB * t_b + 0.5 * T_SE * t_e) / I_Y_W,  # 23
        (-R_W * fx_rf + 0.5 * T_SB * t_b + 0.5 * T_SE * t_e) / I_Y_W,  # 24
        (-R_W * fx_lr + 0.5 * (1.0 - T_SB) * t_b + 0.5 * (1.0 - T_SE) * t_e)
        / I_Y_W,  # 25
        (-R_W * fx_rr + 0.5 * (1.0 - T_SB) * t_b + 0.5 * (1.0 - T_SE) * t_e)
        / I_Y_W,  # 26
        v + A * psip - vuf,  # 27 d(dyf)
        v - B * psip - vur,  # 28 d(dyr)
    )


@wp.kernel
def step_kernel(
    actions: wp.array(dtype=wp.vec2),
    observation: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    state: wp.array(dtype=vec29),
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

    # Input constraints
    vdelta = wp.clamp(actions[i][0], -1.0, 1.0) * STEER_V_MAX
    if (vdelta < 0.0 and s[2] <= STEER_MIN) or (vdelta > 0.0 and s[2] >= STEER_MAX):
        vdelta = 0.0
    accel = wp.clamp(actions[i][1], -1.0, 1.0) * A_MAX
    abs_u = wp.abs(s[3])
    pos_lim = wp.where(abs_u > V_SWITCH, A_MAX * V_SWITCH / wp.max(abs_u, 0.1), A_MAX)
    accel = wp.clamp(accel, -A_MAX, pos_lim)
    if (s[3] <= V_MIN and accel < 0.0) or (s[3] >= V_MAX and accel > 0.0):
        accel = 0.0

    # RK4 substeps (clamp after every substep so wheels don't drift
    #     negative mid-integration and feed Pacejka garbage)
    for _ in range(SUBSTEPS):
        k1 = mb_deriv(s, vdelta, accel)
        k2 = mb_deriv(s + k1 * DT_SUB_HALF, vdelta, accel)
        k3 = mb_deriv(s + k2 * DT_SUB_HALF, vdelta, accel)
        k4 = mb_deriv(s + k3 * DT_SUB, vdelta, accel)
        s = s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * DT_SUB_SIX
        s = vec29(
            s[0],
            s[1],
            wp.clamp(s[2], STEER_MIN, STEER_MAX),
            wp.clamp(s[3], V_MIN, V_MAX),
            s[4],
            s[5],
            s[6],
            s[7],
            s[8],
            s[9],
            s[10],
            s[11],
            s[12],
            s[13],
            s[14],
            s[15],
            s[16],
            s[17],
            s[18],
            s[19],
            s[20],
            s[21],
            s[22],
            wp.max(s[23], 0.0),
            wp.max(s[24], 0.0),
            wp.max(s[25], 0.0),
            wp.max(s[26], 0.0),
            s[27],
            s[28],
        )

    # Crash / reward / waypoint
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
    vx_w = s[3] * ch - s[10] * sh
    vy_w = s[3] * sh + s[10] * ch
    v_along = vx_w * wp.cos(cth) + vy_w * wp.sin(cth)

    progress = wp.max(v_along, -2.0) * PROGRESS_SCALE * DT
    wall = -WALL_PENALTY_COEF * wp.exp(-WALL_PENALTY_RATE * edt_val)
    term_pen = wp.where(term, -TERM_PENALTY, 0.0)
    reward[i] = progress + wall + term_pen

    if term:
        done[i] = DONE_TERMINATED
    elif trunc:
        done[i] = DONE_TRUNCATED
    else:
        done[i] = 0

    # Reset on term/trunc
    if term or trunc:
        rng = wp.rand_init(seed_base + i * 73 + steps * 31)
        rnd = wp.int32(wp.randf(rng) * wp.float32(n_cl)) % n_cl
        rpt = centerline[rnd]
        s = reset29(rpt[0], rpt[1], rpt[2])
        steps = 0
        new_wp = rnd
        ch = wp.cos(s[4])
        sh = wp.sin(s[4])

    # LIDAR raycast on EDT (sphere-tracing)
    lx = s[0] + A * ch
    ly = s[1] + A * sh
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

    # Frenet + lookahead
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

    # Proprioceptive: delta, u, v, psip
    observation[i, 0] = s[2]
    observation[i, 1] = s[3]
    observation[i, 2] = s[10]
    observation[i, 3] = s[5]

    state[i] = s
    step_count[i] = steps
    waypoint_idx[i] = new_wp


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
        # Return in (w, h) layout to match self.edt
        return idx.reshape(h, w).astype(np.int32).T


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

        self.state = wp.zeros(num_envs, dtype=vec29, device=d)
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
        self.state_buf = wp.to_torch(self.state).view(num_envs, 29)
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
    project: str = "f1tenth-mb",
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
