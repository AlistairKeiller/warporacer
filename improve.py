"""Ask Ollama to rewrite main.py shorter/faster, then verify the trained model.

Verification is anti-cheat: the policy's action stream is replayed through a
pure-NumPy reference RK4 with locked physics. Any main.py that weakens the
simulator makes the trained policy fail this replay and get rejected.

Prereq: `ollama serve` running locally; pull a coding model first, e.g.
        `ollama pull qwen3-coder:30b` then run with --model qwen3-coder:30b.
"""

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import torch
import typer

# Locked physics (must equal main.py; verifier uses these, not main.py's).
MU, LF, LR, G = 1.0489, 0.15875, 0.17145, 9.81
A_MAX_CMD, STEER_V_MAX = 9.51, 3.2
WIDTH, LENGTH, DT, SUBSTEPS = 0.31, 0.58, 1 / 60, 6
A_MAX_PHY = MU * G
HALF_DIAG = float(np.hypot(WIDTH / 2, LENGTH / 2))
DT_SUB = DT / SUBSTEPS

ROOT = Path(__file__).parent.resolve()
MAIN = ROOT / "main.py"
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
OLLAMA = "http://localhost:11434/api/chat"

SYSTEM = """You are an expert in PyTorch + NVIDIA Warp + PPO. Rewrite the user's main.py to be SHORTER or FASTER (higher steps/sec).

Hard constraints (verifier rejects violations):
- DO NOT change MU, LF, LR, A_MAX, STEER_*, V_*, WIDTH, LENGTH, DT, G, SUBSTEPS, or the friction-circle clamp logic.
- DO NOT change the observation layout (133-D) or action dim (2).
- KEEP the CLI: `python main.py MAP --num-envs N --iterations I --log-dir D --no-wandb --no-record`.
- KEEP saving log_dir/agent_final.pt with keys: agent, obs_mean, obs_var, obs_count.
- KEEP writing log_dir/result.json with at least: sps, lines_of_code.

Reply with EXACTLY one Python code block in triple-backtick fences, containing the full new main.py. No prose."""


def ask(model: str, code: str, num_ctx: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "# main.py\n```python\n" + code + "\n```"},
            ],
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 64,
            },
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        content = json.loads(r.read())["message"]["content"]
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", content, re.S)
    if not blocks:
        raise ValueError(
            f"no code block in response (first 500 chars):\n{content[:500]}"
        )
    src = max(blocks, key=len)
    ast.parse(src)
    return src


def train(main_path, map_yaml, log_dir, iterations, num_envs, timeout_s):
    log_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            sys.executable,
            str(main_path),
            str(map_yaml),
            "--num-envs",
            str(num_envs),
            "--iterations",
            str(iterations),
            "--log-dir",
            str(log_dir),
            "--no-wandb",
            "--no-record",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(main_path.parent),
    )
    (log_dir / "stdout.log").write_text(r.stdout)
    (log_dir / "stderr.log").write_text(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"main.py rc={r.returncode}: {r.stderr[-500:]}")
    return json.loads((log_dir / "result.json").read_text())


def verify(main_path, ckpt, map_yaml, max_steps=4000, friction_tol=0.05):
    spec = importlib.util.spec_from_file_location(f"cm_{ckpt.parent.name}", main_path)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ckpt, map_location=dev)
    env = M.RacingEnv(map_yaml, num_envs=1, seed=42)
    agent = M.Agent().to(env.device)
    agent.load_state_dict(blob["agent"])
    agent.eval()
    obs_mean = blob["obs_mean"].to(env.device)
    inv_std = torch.rsqrt(blob["obs_var"].to(env.device) + 1e-8)

    raw, _ = env.reset()
    init = env.cars_buf[0].cpu().numpy()[:5].astype(np.float64)
    actions = []
    obs = ((raw - obs_mean) * inv_std).clamp(-10, 10)
    with torch.no_grad():
        for _ in range(max_steps):
            a = agent.deterministic(obs)
            actions.append(a[0].cpu().numpy())
            raw, _, term, trunc, _ = env.step(a)
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
            obs = ((raw - obs_mean) * inv_std).clamp(-10, 10)

    # Reference RK replay with locked physics.
    s = init.copy()
    cl, dt_map = env.map.centerline, env.map.dt
    n_cl, res, ox, oy, h_pix = len(cl), env.map.res, env.map.ox, env.map.oy, env.map.h
    last_wp = int(np.argmin(((cl - s[:2]) ** 2).sum(1)))
    progress = laps = fric_viol = 0
    collided = False
    for a_norm in actions:
        sv = float(np.clip(a_norm[0], -1, 1)) * STEER_V_MAX
        a_long_cmd = float(np.clip(a_norm[1], -1, 1)) * A_MAX_CMD
        for _ in range(SUBSTEPS):
            _, _, d, v, p = s
            d_psi = float(
                np.clip(
                    v * np.tan(d) / (LF + LR),
                    -A_MAX_PHY / max(abs(v), 0.5),
                    A_MAX_PHY / max(abs(v), 0.5),
                )
            )
            a_lat = v * d_psi
            a_long_max = float(np.sqrt(max(A_MAX_PHY**2 - a_lat**2, 0.0)))
            if abs(a_long_cmd) > a_long_max + 0.01:
                fric_viol += 1
            a_long = float(np.clip(a_long_cmd, -a_long_max, a_long_max))
            s = s + DT_SUB * np.array(
                [v * np.cos(p), v * np.sin(p), 0.0, a_long, d_psi]
            )
            s[2] = float(np.clip(s[2] + sv * DT_SUB, -0.4189, 0.4189))
            s[3] = float(np.clip(s[3], -5.0, 20.0))
        px = int(np.clip((s[0] - ox) / res, 0, dt_map.shape[1] - 1))
        py = int(np.clip(h_pix - 1 - (s[1] - oy) / res, 0, dt_map.shape[0] - 1))
        if float(dt_map[py, px]) * res < HALF_DIAG:
            collided = True
            break
        wp_i = int(np.argmin(((cl - s[:2]) ** 2).sum(1)))
        d_wp = wp_i - last_wp
        if d_wp > n_cl // 2:
            d_wp -= n_cl
        elif d_wp < -n_cl // 2:
            d_wp += n_cl
        progress += d_wp
        if progress >= n_cl:
            laps += 1
            progress -= n_cl
        last_wp = wp_i

    rate = fric_viol / max(len(actions) * SUBSTEPS, 1)
    return {
        "pass": bool((laps >= 1) and (not collided) and (rate < friction_tol)),
        "laps": int(laps),
        "collided": bool(collided),
        "friction_violation_rate": float(rate),
        "steps": int(len(actions)),
    }


def main(
    map_yaml: Path,
    iterations: int = 10,
    train_iters: int = 1000,
    num_envs: int = 4096,
    train_timeout_s: int = 7200,
    verify_steps: int = 4000,
    friction_tol: float = 0.05,
    model: str = "qwen3-coder:30b",
    num_ctx: int = 65_536,
):
    base = RUNS / "baseline"
    if not (base / "agent_final.pt").exists():
        train(MAIN, map_yaml, base, train_iters, num_envs, train_timeout_s)
    bres = json.loads((base / "result.json").read_text())
    bv = verify(MAIN, base / "agent_final.pt", map_yaml, verify_steps, friction_tol)
    best_lines = bres.get("lines_of_code", len(MAIN.read_text().splitlines()))
    best_sps = bres.get("sps", 0)
    best_pass = bv["pass"]
    print(f"[baseline] pass={best_pass} lines={best_lines} sps={best_sps} {bv}")
    if not best_pass:
        print("  hint: increase --train-iters or relax --friction-tol")

    for it in range(iterations):
        cd = RUNS / f"iter_{it:03d}"
        cd.mkdir(exist_ok=True)
        try:
            new = ask(model, MAIN.read_text(), num_ctx)
            (cd / "main.py").write_text(new)
            tres = train(
                cd / "main.py", map_yaml, cd, train_iters, num_envs, train_timeout_s
            )
            v = verify(
                cd / "main.py",
                cd / "agent_final.pt",
                map_yaml,
                verify_steps,
                friction_tol,
            )
        except Exception as e:
            print(f"[{it}] failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        lines = tres.get("lines_of_code", len(new.splitlines()))
        sps = tres.get("sps", 0)
        better = v["pass"] and (
            not best_pass or lines < best_lines or sps > best_sps * 1.05
        )
        tag = "ADOPT" if better else ("PASS" if v["pass"] else "FAIL")
        print(
            f"[{it:03d}] {tag} lines={lines} sps={sps} laps={v['laps']} "
            f"fric={v['friction_violation_rate']:.3f} collided={v['collided']}"
        )
        if better:
            shutil.copy2(MAIN, cd / "main.py.replaced")
            shutil.copy2(cd / "main.py", MAIN)
            best_lines = min(best_lines, lines)
            best_sps = max(best_sps, sps)
            best_pass = True


if __name__ == "__main__":
    typer.run(main)
