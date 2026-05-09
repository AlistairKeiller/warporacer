from pathlib import Path

import torch
import numpy as np
import warp as wp
from typer import run

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
    iterations: int = 2000,
    record_every: int = 100,
    record_steps: int = 2000,
    device: str = None,
    use_wandb: bool = True,
    log_dir: Path = Path("./logs"),
):
    if interactive:
        num_envs = 1

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
        env = Environment(map_yaml, num_envs, seed)

        if interactive:
            env.vs.interactive_render_loop()
        else:
            print("TODO")
            pass

if __name__ == "__main__":
    #run(main)
    main()