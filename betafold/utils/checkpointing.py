"""
激活检查点是一种以计算时间换内存的技术。通常情况下, 前向传递时的激活值会被保存, 
用于反向传播期间的梯度计算。而检查点区域中的前向计算省略了为反向传播保存激活值, 而是在反向传播期间重新计算它们。
具体而言, 在前向传递中，函数将以 torch.no_grad() 的方式运行, 即不保存中间激活值。
计算速度会变慢, 内存压力变小。
"""
from importlib.util import find_spec
from typing import Any, Tuple, List, Callable, Optional

deepspeed_is_installed = find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

import torch
import torch.utils.checkpoint


BLOCK_ARG = Any
BLOCK_ARGS = Tuple[BLOCK_ARG]


def get_checkpoint_fn() -> Callable:
    deepspeed_is_configured = (
        deepspeed_is_installed and
        deepspeed.checkpointing.is_configured()
    )
    if deepspeed_is_configured:
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint

    return checkpoint


def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.
    使用激活检查点的方式对一系列块进行分块，并运行每个块。
    在这里，我们将“块”定义为其唯一输入是前一个块的输出的可调用对象。

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer 
            checkpoints, and trades memory for speed. If None, no checkpointing 
            is performed.
            每个检查点包含的block数量。
            #? 较高的值对应较少的检查点, 此时内存的消耗理应更小, 速度理应更慢。但是此处原文含义是相反的？
            如果为 None, 则不执行检查点。
    Returns:
        The output of the final block
    """
    # region: 定义一些工具函数
    # 确保参数是元组形式
    def wrap(a):
        return (a,) if type(a) is not tuple else a

    # 逐个执行块
    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    # 分割器函数，用于根据起始和结束索引执行一部分块
    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced
    # endregion

    # 当块的参数只有一个时，避免出现错误
    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)
    
    # 如果不执行检查点或者梯度未启用（比如推理），则直接执行所有块
    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    #* checkpoint函数(以torch.utils.checkpoint.checkpoint为例), 其实就是对于传入的函数只计算输入和输出的激活值。
    #* 将多个模块包含在单个检查点中, 意味着中间的激活值不会被计算
    checkpoint = get_checkpoint_fn()
    # 分块运行并执行激活检查点
    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args)
        args = wrap(args)

    return args
