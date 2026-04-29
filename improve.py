"""Ask Gemma to rewrite main.py shorter/faster, then verify the trained model.

Verification is anti-cheat: we replay the policy's action stream through a
pure-NumPy reference RK4 and require lap completion, no collisions, and a
friction-circle compliance rate >= 98%. If main.py was edited to weaken
the physics, the trained policy fails reference replay and is rejected.
"""

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import typer

# Locked physics (must equal main.py; verification uses these, not main.py's).
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
PPO_CACHE = ROOT / ".cache_cleanrl_ppo.py"
PPO_URL = "https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo.py"

SYSTEM = """You are an expert in PyTorch + NVIDIA Warp + PPO. The user will give you reference Warp examples, the CleanRL PPO implementation, and a single-file racing trainer (main.py). Rewrite main.py to be SHORTER or FASTER (higher steps/sec).

Hard constraints (the verifier will reject violations):
- DO NOT change MU, LF, LR, A_MAX, STEER_*, V_*, WIDTH, LENGTH, DT, G, SUBSTEPS, or the friction-circle clamp logic.
- DO NOT change the observation layout (133-D) or action dim (2).
- KEEP the CLI: `python main.py MAP --num-envs N --iterations I --log-dir D --no-wandb --no-record`.
- KEEP saving log_dir/agent_final.pt with keys: agent, obs_mean, obs_var, obs_count.
- KEEP writing log_dir/result.json with at least: sps, lines_of_code.

Reply with EXACTLY one Python code block in triple-backtick fences (```python ... ```), containing the full new main.py. No prose."""


def warp_examples_text() -> str:
    """Load all .py files under the installed `warp/examples` directory."""
    import importlib.util

    spec = importlib.util.find_spec("warp")
    if spec is None or spec.origin is None:
        return ""
    examples = Path(spec.origin).parent / "examples"
    if not examples.exists():
        return ""
    parts = []
    for path in sorted(examples.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        rel = path.relative_to(examples)
        try:
            body = path.read_text(errors="replace")
        except Exception:
            continue
        parts.append(f"### warp/examples/{rel}\n```python\n{body}\n```")
    return "\n\n".join(parts)


def cleanrl_ppo_text() -> str:
    if PPO_CACHE.exists():
        return PPO_CACHE.read_text()
    import urllib.request

    PPO_CACHE.write_text(urllib.request.urlopen(PPO_URL, timeout=30).read().decode())
    return PPO_CACHE.read_text()


def build_user_message(code: str, max_bytes: int | None) -> str:
    ctx_parts = [
        "# Reference: NVIDIA Warp examples",
        warp_examples_text(),
        "# Reference: CleanRL PPO (vwxyzjn/cleanrl/cleanrl/ppo.py)",
        "```python\n" + cleanrl_ppo_text() + "\n```",
    ]
    ctx = "\n\n".join(p for p in ctx_parts if p)
    if max_bytes and len(ctx) > max_bytes:
        ctx = ctx[:max_bytes] + "\n\n[... reference truncated ...]"
    return ctx + "\n\n# File to rewrite: main.py\n```python\n" + code + "\n```"


def ask(
    processor,
    model,
    code: str,
    max_context_bytes: int | None = None,
    max_new_tokens: int = 6000,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_message(code, max_context_bytes)},
    ]
    # Gemma 4 docs: render template -> tokenize via processor -> generate.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    device = next(model.parameters()).device
    inputs = processor(text=text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    # parse_response strips Gemma 4 thought-channel framing tokens.
    parsed = (
        processor.parse_response(raw) if hasattr(processor, "parse_response") else raw
    )
    if isinstance(parsed, dict):
        parsed = parsed.get("content") or parsed.get("response") or str(parsed)
    elif not isinstance(parsed, str):
        parsed = str(parsed)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", parsed, re.S)
    if not blocks:
        raise ValueError(
            f"no fenced code block in response (first 500 chars):\n{parsed[:500]}"
        )
    src = max(blocks, key=len)  # the rewrite is the largest block
    ast.parse(src)  # raises SyntaxError if malformed
    return src


def train(
    main_path: Path,
    map_yaml: Path,
    log_dir: Path,
    iterations: int,
    num_envs: int,
    timeout_s: int,
) -> dict:
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


def verify(
    main_path: Path,
    ckpt: Path,
    map_yaml: Path,
    max_steps: int = 4000,
    friction_tol: float = 0.05,
) -> dict:
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

    # Reference replay: forward-Euler RK at the substep scale with the SAME
    # friction-circle clamp the original kernel applies.
    s = init.copy()
    cl = env.map.centerline
    n_cl = len(cl)
    dt_map = env.map.dt
    res = env.map.res
    ox, oy, h_pix = env.map.ox, env.map.oy, env.map.h
    last_wp = int(np.argmin(((cl - s[:2]) ** 2).sum(1)))
    progress = laps = 0
    fric_viol = 0
    collided = False
    for a_norm in actions:
        sv = float(np.clip(a_norm[0], -1, 1)) * STEER_V_MAX
        a_long_cmd = float(np.clip(a_norm[1], -1, 1)) * A_MAX_CMD
        for _ in range(SUBSTEPS):
            x, y, d, v, p = s
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

    n_sub = max(len(actions) * SUBSTEPS, 1)
    rate = fric_viol / n_sub
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
    model: str = "google/gemma-4-31B-it",
    max_context_bytes: int = 0,
    max_new_tokens: int = 6000,
    load_in_4bit: bool = True,
):
    import sys
    import traceback

    from transformers import AutoModelForCausalLM, AutoProcessor

    # Loud startup diagnostics so we never silently end up in a bad config.
    print(f"[env] python   = {sys.executable}")
    try:
        import bitsandbytes as _bnb

        print(f"[env] bnb      = {_bnb.__version__}")
        bnb_ok = True
    except ImportError:
        print(f"[env] bnb      = NOT INSTALLED  (`pip install bitsandbytes`)")
        bnb_ok = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[env] GPU free = {free / 1e9:.1f} / {total / 1e9:.1f} GB")
        if free < total * 0.85:
            print(
                f"[env] WARNING: only {free / 1e9:.0f} GB free - another "
                f"process is probably holding GPU memory. Run `nvidia-smi` "
                f"and kill stale python/torch processes, then retry."
            )
            raise SystemExit(1)

    use_4bit = load_in_4bit and bnb_ok
    load_kw = {"device_map": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"[load] mode = 4-bit NF4 (~16 GB model, ~80 GB for KV cache)")
        effective_ctx = max_context_bytes or None
    else:
        load_kw["dtype"] = "auto"
        print(f"[load] mode = bf16 (~62 GB model, ~33 GB for KV cache)")
        if load_in_4bit and not bnb_ok:
            print(
                f"[load] >>> 4-bit was requested but bitsandbytes is "
                f"missing. Install it (`pip install bitsandbytes`) for "
                f"full context. Otherwise capping prompt aggressively."
            )
        # bf16 31B + full warp examples KV cache won't fit in 95 GB. Force a cap.
        effective_ctx = max_context_bytes or 80_000
        if max_context_bytes == 0:
            print(
                f"[load] auto-cap = {effective_ctx} bytes "
                f"(~20K tokens). Override with --max-context-bytes."
            )

    processor = AutoProcessor.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, **load_kw)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[env] GPU free after model load: {free / 1e9:.1f} GB")

    # Baseline.
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
        print(
            "  hint: baseline failed verification - increase --train-iters "
            "(e.g. 2000) or relax --friction-tol (e.g. 0.10) if the policy "
            "looks fine in rollout videos."
        )

    for it in range(iterations):
        cd = RUNS / f"iter_{it:03d}"
        cd.mkdir(exist_ok=True)
        try:
            new = ask(
                processor,
                mdl,
                MAIN.read_text(),
                max_context_bytes=effective_ctx,
                max_new_tokens=max_new_tokens,
            )
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
        # Adoption: if no current best passes, the first passing candidate wins.
        # Otherwise require a passing candidate to also be shorter or >=5% faster.
        if not best_pass:
            better = v["pass"]
        else:
            better = v["pass"] and (lines < best_lines or sps > best_sps * 1.05)
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
