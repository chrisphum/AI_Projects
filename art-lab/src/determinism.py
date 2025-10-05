import os, random, torch, numpy as np

def make_deterministic(seed:int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS notes: no cudnn, but set flags anyway for CPU fallback parity
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # harmless on Mac
    # Diffusers/samplers may still have nondet ops; we’ll gate-test below.