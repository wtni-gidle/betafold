import torch

def is_fp16_enabled():
    """
    检查当前是否启用了混合精度训练(FP16).

    Returns:
        bool: 如果启用了FP16, 则返回True; 否则返回False。
    """
    # Autocast world
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled
