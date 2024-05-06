from functools import partial

import torch
import torch.nn as nn
from typing import Tuple, Optional

from betafold.model.primitives import Linear, LayerNorm

from betafold.utils.tensor_utils import add, one_hot, tensor_tree_map, dict_multimap, add_first_dims

# alphafold版本的InputEmbedder
# target_feat和residue_index生成的二维位置编码相加得到pair repr.
# msa_feat和target_feat相加得到msa repr.
class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor) -> torch.Tensor:
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N_res]
        Returns:
            [*, N_res, N_res, c_z]
        """
        # [*, N_res, N_res]
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        # [*, N_res, N_res, no_bins]
        d = one_hot(d, boundaries)
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Dict containing
                "target_feat":
                    Features of shape [*, N_res, tf_dim]
                "residue_index":
                    Features of shape [*, N_res]
                "msa_feat":
                    Features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(
            pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(
            pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb

# drfold版本的InputEmbedder
# target_feat和二维位置编码相加得到pair repr.
# msa_feat和target_feat和一维位置编码相加得到msa repr.
class InputEmbedder2(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_m: int,
        c_z: int,
        max_len_seq: int,
        no_pos_bins_1d: int,
        pos_wsize_2d: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_m:
                MSA embedding dimension
            c_z:
                Pair embedding dimension
            max_len_seq:
                序列长度不能超过这个值
            no_pos_bins_1d:
                一维位置编码的维度
            pos_wsize_2d:
                二维位置编码的窗口大小
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_m = c_m
        self.c_z = c_z

        # RPE stuff
        self.max_seq_len = max_len_seq
        self.no_pos_bins_1d = no_pos_bins_1d
        self.pos_wsize_2d = pos_wsize_2d
        self.no_pos_bins_2d = 2 * self.pos_wsize_2d + 1

        self.linear_tf_m = Linear(self.tf_dim, self.c_m)
        self.linear_msa_m = Linear(self.msa_dim, self.c_m)
        self.linear_pos_1d = Linear(self.no_pos_bins_1d, self.c_m)
        self.linear_tf_z_i = Linear(self.tf_dim, self.c_z)
        self.linear_tf_z_j = Linear(self.tf_dim, self.c_z)
        self.linear_pos_2d = Linear(self.no_pos_bins_2d, self.c_z)

        self.pos_1d = self.compute_pos_1d()
        self.pos_2d = self.compute_pos_2d()

    def compute_pos_1d(self) -> torch.Tensor:
        """
        return: [max_seq_len, no_pos_bins_1d]的二进制矩阵。即每一个位置用二进制表示
        e.g. 
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        """
        pos = torch.arange(self.max_seq_len)
        rel_pos = ((pos[:,None] & (1 << torch.arange(self.no_pos_bins_1d)))) > 0
        
        return rel_pos.float()

    def compute_pos_2d(self) -> torch.Tensor:
        """
        return: [max_seq_len, max_seq_len, no_pos_bins_2d]
        """
        pos = torch.arange(self.max_seq_len)
        rel_pos = (pos[None, :] - pos[:, None]).clamp(-self.pos_wsize_2d, self.pos_wsize_2d)
        rel_pos_enc = nn.functional.one_hot(rel_pos + self.pos_wsize_2d, self.no_pos_bins_2d)
        
        return rel_pos_enc.float()

    def forward(
        self,
        tf: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tf: [*, N_res, tf_dim]
        msa: [*, N_seq, N_res, msa_dim]

        return:
        msa_emb: [*, N_seq, N_res, c_m]
        pair_emb: [*, N_res, N_res, c_z]
        """
        n_res = tf.shape[-2]
        # device
        self.pos_1d = self.pos_1d.to(tf.device, tf.dtype)
        self.pos_2d = self.pos_2d.to(tf.device, tf.dtype)

        # [*, N_res, c_m]
        tf_emb_m = self.linear_tf_m(tf)
        # [*, N_seq, N_res, c_m]
        msa_emb_m = self.linear_msa_m(msa)
        # [N_res, c_m] -> [*, N_seq, N_res, c_m]
        pos_enc_1d = self.linear_pos_1d(self.pos_1d[:n_res])
        pos_enc_1d = add_first_dims(pos_enc_1d, len(msa.shape) - 2)
        # [*, N_seq, N_res, c_m]
        msa_emb = msa_emb_m + tf_emb_m[..., None, :, :] + pos_enc_1d

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        # [N_res, N_res, c_z] -> [*, N_res, N_res, c_z]
        pair_emb = self.linear_pos_2d(self.pos_2d[:n_res, :n_res])
        pair_emb = add_first_dims(pair_emb, len(tf.shape) - 2)
        # [*, N_res, N_res, c_z]
        pair_emb = add(
            pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(
            pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )

        return msa_emb, pair_emb

# drfold版本的SSEmbedder
# 全连接层
class SSEmbedder(nn.Module):
    def __init__(
        self, 
        ss_dim: int, 
        c_z: int,
        **kwargs,
    ):
        """
        Args:
            ss_dim:
                Final dimension of the ss features
            c_z:
                Pair embedding dimension
        """
        super().__init__()
        self.ss_dim = ss_dim
        self.c_z = c_z
        self.ss_linear = Linear(self.ss_dim, self.c_z)

    def forward(self, ss: torch.Tensor) -> torch.Tensor:
        """
        ss: [*, N_res, N_res, ss_dim]

        return:
        pair_emb_ss: [*, N_res, N_res, c_z]
        """
        pair_emb_ss = self.ss_linear(ss)
        
        return pair_emb_ss

# alphafold版本的RecyclingEmbedder
class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N_res, C_m]
        m_update = self.layer_norm_m(m)
        if inplace_safe:
            #* 这里要确保m在后面不会被用到
            m.copy_(m_update)
            m_update = m

        # [*, N_res, N_res, C_z]
        z_update = self.layer_norm_z(z)
        if inplace_safe:
            #* 这里要确保z在后面不会被用到
            z.copy_(z_update)
            z_update = z

        #* 此处实现alphafold原文的距离嵌入, 而非openfold的方法
        #* alphafold: 使用one_hot, 指定的boundaries实际上就是区间的中点
        #* openfold: 常规的查看距离落在哪个区间, 但是可能不落在任何一个区间, 感觉有点问题
        #* 以后更推荐用one_hot方法, 即看离哪个区间的中点最近。这种方法更简单，且没有bug
        # This squared method might become problematic in FP16 mode.
        boundaries = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
        )
        boundaries = boundaries ** 2
        # [*, N_res, N_res]
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1
        )
        # [*, N_res, N_res, no_bins]
        d = one_hot(d, boundaries)

        # [*, N_res, N_res, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update


def fourier_encode_dist(
    x: torch.Tensor, 
    num_encodings: int = 20, 
    include_self: bool = True
) -> torch.Tensor:
    # from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x

# drfold版本的recycling, 区别只是距离编码方式不同
class RecyclingEmbedder2(nn.Module):
    def __init__(
        self, 
        c_m: int, 
        c_z: int, 
        dis_encoding_dim: int
    ) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.dis_encoding_dim = dis_encoding_dim
        
        self.linear = Linear(self.dis_encoding_dim * 2 + 1, c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)
    
    def forward(
        self, 
        m: torch.Tensor, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        m: 
            [*, N_res, c_m] First row of the MSA embedding.
        z: 
            [*, N_res, N_res, c_z] Pair embedding.
        x: 
            [*, N_res, 3] predicted N coordinates

        return:
            m_update: [*, N_res, c_m]
            z_update: [*, N_res, N_res, c_z]
        """
        # [*, N_res, C_m]
        m_update = self.layer_norm_m(m)
        if inplace_safe:
            #* 这里要确保m在后面不会被用到
            m.copy_(m_update)
            m_update = m

        # [*, N_res, N_res, C_z]
        z_update = self.layer_norm_z(z)
        if inplace_safe:
            #* 这里要确保z在后面不会被用到
            z.copy_(z_update)
            z_update = z
        
        # [*, N_res, N_res]
        d = (x[..., None, :] - x[..., None, :, :]).norm(dim = -1)
        d = fourier_encode_dist(d, self.dis_encoding_dim)

        z_update = add(
            z_update, 
            self.linear(d), 
            inplace_safe
        )

        return m_update, z_update


#todo template, preembedding没写