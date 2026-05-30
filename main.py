from pathlib import Path

import torch
import numpy as np
import warp as wp
from typer import run
import os
import wandb

from include.agent import Agent, train, record_rollout
from include.constants import *
from include.environment import Environment

# If you see this error and you have more than one GPU (iGPU & eGPU):
#   "Warp UserWarning: Could not register GL buffer since CUDA/OpenGL interoperability is not available.
#   Falling back to copy operations between the Warp array and the OpenGL buffer."
# Then you have to make sure ALL aspects of the Python program is running on GPU. On Windows you find
# the Python executable and set to "High Performance" in Windows Graphics settings.

def main(
    map_yaml: Path =  Path(".\\maps\\berlin.yaml"),
    num_envs: int = 1024,
    seed: int = 0,
    interactive: bool = True,
    live_viewer: bool = False,
    iterations: int = 2000,
    record_every: int = 100,
    record_steps: int = 2000,
    device: str = None,
    use_wandb: bool = False,
    log_dir: Path = Path("./logs"),
):
    
    # Interactive mode overrides
    if interactive:
        num_envs = 1
        live_viewer = True

    if not device:
        device = wp.get_device()

    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with wp.ScopedDevice(device):
        env = Environment(map_yaml, num_envs, seed, live_viewer)

        if interactive:
            env.vs.interactive_render_loop()
        else:
            if use_wandb:
                try:
                    wandb.init(
                        project="warporacer",
                        name=f"seed{seed}_n{num_envs}",
                        config={
                            "num_envs": num_envs,
                            "iterations": iterations,
                            "seed": seed,
                            "map": str(map_yaml),
                        },
                    )
                except Exception as e:
                    print(f"[WandB] Init failed: {e}")
                    
            agent = torch.compile(Agent(obs_dim=OBS_DIM).to(str(env.device)))
            elapsed, obs_rms, ret_rms, step = train(
                env,
                agent,
                iterations=iterations,
                log_dir=log_dir,
                record_every=record_every,
                record_steps=record_steps,
                use_wandb_train=use_wandb
            )

            print(f"[Done!] {elapsed:.1f}s")

            torch.save(
                {
                    "agent": agent.state_dict(),
                    "obs_mean": obs_rms.mean.cpu(),
                    "obs_var": obs_rms.var.cpu(),
                    "obs_count": obs_rms.count,
                },
                log_dir / "agent_final.pt",
            )

            print(f"[Saved!]")

            out = log_dir / "rollout_final.mp4"
            record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)

            if use_wandb:
                try:
                    wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=step)
                except Exception as e:
                    print(f"[WandB] Final rollout video failed: {e}")

if __name__ == "__main__":
    run(main)

# Notes:
# main(interactive=False, num_envs=16384, map_yaml=Path(".//maps//berlin.yaml"))
# if os.name == "nt":
#     main(interactive=False, num_envs=1024, map_yaml=Path(".//maps//berlin.yaml"))
# else:
#     main(interactive=False, num_envs=1024, map_yaml=Path("./maps/berlin.yaml"))
# env.vs.render() # Live rendering of training