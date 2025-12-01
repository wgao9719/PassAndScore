import torch
from typing import Optional, Union
from light_malib.utils.logger import Logger


def resolve_device(
    preferred: Optional[Union[str, torch.device]] = None, use_cuda: bool = False
):
    """Return a torch.device, falling back gracefully when backends are unavailable."""
    name = None
    if isinstance(preferred, torch.device):
        name = str(preferred)
    elif isinstance(preferred, str) and preferred:
        name = preferred
    else:
        name = "cuda" if use_cuda else "cpu"

    if ":" in name:
        base, suffix = name.split(":", 1)
        base = base.lower()
        name = f"{base}:{suffix}"
    else:
        base = name.lower()
        name = base
    base = name.split(":")[0]

    if base == "cuda":
        if not torch.cuda.is_available():
            Logger.warning("CUDA requested but not available; falling back to CPU.")
            name = "cpu"
            base = "cpu"
    elif base == "mps":
        if not torch.backends.mps.is_available():
            Logger.warning("MPS requested but not available; falling back to CPU.")
            name = "cpu"
            base = "cpu"
    else:
        # allow explicit "cpu" or other torch-recognized strings
        pass

    return torch.device(name)
