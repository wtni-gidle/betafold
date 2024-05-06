"""
superimpose two structures, 但是建议只用于validation
一方面可能不是最优对齐, 另一方面这里是把mask的部分去了再做的对齐。mask的坐标被设置为0
"""
from typing import Tuple
import numpy as np
import torch

from Bio.SVDSuperimposer import SVDSuperimposer


def _superimpose_np(
    reference: np.ndarray, 
    coords: np.ndarray
) -> Tuple[np.ndarray, np.float64]:
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [N, 3] reference array
            coords:
                [N, 3] array
        Returns:
            A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()


def _superimpose_single(
    reference: torch.Tensor, 
    coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference_np = reference.detach().cpu().numpy()    
    coords_np = coords.detach().cpu().numpy()
    superimposed, rmsd = _superimpose_np(reference_np, coords_np)
    return coords.new_tensor(superimposed), coords.new_tensor(rmsd)


def superimpose(
    reference: torch.Tensor, 
    coords: torch.Tensor, 
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [*, N, 3] reference tensor
            coords:
                [*, N, 3] tensor
            mask:
                [*, N] tensor
        Returns:
            A tuple of [*, N, 3] superimposed coords and [*] final RMSDs.
    """
    def select_unmasked_coords(coords, mask):
        return torch.masked_select(
            coords,
            (mask > 0.)[..., None],
        ).reshape(-1, 3)

    batch_dims = reference.shape[:-2]
    # [*, N, 3]
    flat_reference = reference.reshape((-1,) + reference.shape[-2:])
    flat_coords = coords.reshape((-1,) + reference.shape[-2:])
    flat_mask = mask.reshape((-1,) + mask.shape[-1:])
    superimposed_list = []
    rmsds = []
    for r, c, m in zip(flat_reference, flat_coords, flat_mask):
        # 只取出没有被mask的部分
        r_unmasked_coords = select_unmasked_coords(r, m)
        c_unmasked_coords = select_unmasked_coords(c, m)
        superimposed, rmsd = _superimpose_single(
            r_unmasked_coords, 
            c_unmasked_coords
        )

        # This is very inelegant, but idk how else to invert the masking
        # procedure.
        # 将unmask的superimposition填充成原来的大小
        count = 0
        # [N, 3]
        superimposed_full_size = torch.zeros_like(r)
        for i, unmasked in enumerate(m):
            if unmasked:
                superimposed_full_size[i] = superimposed[count]
                count += 1

        superimposed_list.append(superimposed_full_size)
        rmsds.append(rmsd)
    # [* x N, 3]
    superimposed_stacked = torch.stack(superimposed_list, dim=0)
    rmsds_stacked = torch.stack(rmsds, dim=0)
    # [*, N, 3]
    superimposed_reshaped = superimposed_stacked.reshape(
        batch_dims + coords.shape[-2:]
    )
    rmsds_reshaped = rmsds_stacked.reshape(
        batch_dims
    )

    return superimposed_reshaped, rmsds_reshaped
