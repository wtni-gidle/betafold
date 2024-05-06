"""metrics for evaluation"""
"""如果要用做验证集指标，还需要将结果取平均"""
#! 目前这些指标计算只能用于验证集。按照公式计算，和实际软件跑出来的存在区别。
#! 这里只使用每个残基的单原子（比如openfold使用Cα，nucleotide一般用C3'或C4'）
#! 输出的是每个样本的分数
#todo openstructure貌似包含了我想要的关于蛋白质、核酸的原子的约束条件，可以留意一下
from typing import Optional
from functools import partial
import numpy as np
import torch


def drmsd(
    structure_1: torch.Tensor, 
    structure_2: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    structure_1: [*, N, 3]
    structure_2: [*, N, 3]
    mask: [*, N]
    return:
        drmsd: [*]
    """
    if mask is None:
        mask = structure_1.new_ones(structure_1.shape[:-1])

    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d ** 2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    # [*, N, N]
    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = (d1 - d2) ** 2
    drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    # [*]
    drmsd = torch.sum(drmsd, dim=(-1, -2))

    # [n1, ..., n2]
    n = torch.sum(mask, dim=-1)
    # 对于<=1的设置为inf, 这样drmsd就是0
    n = torch.where(n > 1, n, torch.inf)

    drmsd = drmsd * (1 / (n * (n - 1)))
    #! 注意这里openfold原先也是取sum，后来加了min操作，如下所示。理由是Fix batched finetuning bugs
    #! 具体原因不详。一种推测是如果不取min，下面的判断语句会出错。（会犯这种低级错误？）
    #! 也有可能别的考虑，具体训练时再试试看
    # n = d1.shape[-1] if mask is None else torch.min(torch.sum(mask, dim=-1))
    # drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(
    structure_1: np.ndarray, 
    structure_2: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> torch.Tensor:
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if mask is not None:
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(
    p1: torch.Tensor, 
    p2: torch.Tensor, 
    cutoffs: list[float], 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    需要输入已经对齐的结构
    p1: [*, N, 3]
    p2: [*, N, 3]
    mask: [*, N]
    return: [*]
    """
    if mask is None:
        mask = p1.new_ones(p1.shape[:-1])
    # [*]
    n = torch.sum(mask, dim=-1)
    
    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1))
    scores = []
    for c in cutoffs:
        # [*]
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        # score = torch.mean(score)
        scores.append(score)
    # [4, *]
    scores = torch.stack(scores, dim=0)
    # [*]
    scores = torch.mean(scores, dim=0)

    return scores

gdt_ts = partial(gdt, cutoffs=[1., 2., 4., 8.])
gdt_ha = partial(gdt, cutoffs=[0.5, 1., 2., 4.])


def tm_score(
    p1: torch.Tensor, 
    p2: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    需要输入已经对齐的结构
    p1: [*, N, 3]
    p2: [*, N, 3]
    mask: [*, N]
    return: [*]
    """
    if mask is None:
        mask = p1.new_ones(p1.shape[:-1])
    # [*]
    n = torch.sum(mask, dim=-1)
    
    p1 = p1.float()
    p2 = p2.float()
    # [*, N]
    distance_squared = torch.sum((p1 - p2)**2, dim=-1)
    # [*]
    clipped_n = torch.clamp(n, min=19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8
    # [*, N]
    di = 1.0 / (1 + distance_squared / (d0 ** 2))
    # [*]
    scores = torch.sum(di * mask, dim=-1) / n

    return scores


def lddt(
    ref_pos: torch.Tensor, 
    pred_pos: torch.Tensor, 
    mask: Optional[torch.Tensor] = None,
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    残基index距离阈值r为1
    ref_pos: [*, N, 3]
    pred_pos: [*, N, 3]
    mask: [*, N]
    return: [*]
    """
    if mask is None:
        mask = ref_pos.new_ones(ref_pos.shape[:-1])
    
    n = mask.shape[-1] # N
    # [*, N, N]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                ref_pos[..., None, :]
                - ref_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                pred_pos[..., None, :]
                - pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    # [*, N, N]
    dists_to_score = (
        (dmat_true < cutoff) 
        * mask[..., None] 
        * mask[..., None, :]
        * (1.0 - torch.eye(n, device=mask.device))
    )
    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25
    dims = (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score
