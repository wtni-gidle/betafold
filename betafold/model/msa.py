from functools import partial
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from betafold.model.primitives import (
    Linear, 
    LayerNorm,
    Attention, 
    GlobalAttention, 
)
from betafold.utils.chunk_utils import chunk_layer
from betafold.utils.tensor_utils import permute_final_dims


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        pair_bias: bool = False,
        c_z: Optional[int] = None,
        inf: float = 1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )

    def _chunk(
        self, 
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        flash_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def fn(m, biases, flash_mask):
            m = self.layer_norm_m(m)
            return self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=flash_mask,
            )

        inputs = {"m": m}
        if biases is not None:
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)
        if use_flash and flash_mask is not None:
            inputs["flash_mask"] = flash_mask
        else:
            fn = partial(fn, flash_mask=None)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理z和mask
        Args:
            m: 
                [*, N_seq, N_res, C_m]
            z: 
                [*, N_res, N_res, C_z]
            mask: 
                [*, N_seq, N_res]
        Returns:
            m:
                [*, N_seq, N_res, C_m]
            mask_bias:
                [*, N_seq, 1, 1, N_res]
            z:
                [*, 1, H, N_res, N_res]
        """
        # region: mask_bias
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        # endregion

        # region: z
        if (self.pair_bias and
            z is not None and
            self.layer_norm_z is not None and
            self.linear_z is not None
        ):
            chunks = []
            # 在第一个N_res的维度上分块运行，因为Linear和LayerNorm的性质
            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i: i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)
            
                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)
            
            z = torch.cat(chunks, dim=-3)
            
            # [*, 1, H, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)
        # endregion

        return m, mask_bias, z

    def forward(
        self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.    
        """
        # region: biases
        if use_flash:
            # flash不支持带pair bias，只支持mask
            assert z is None
            biases = None
        else:    
            m, mask_bias, z = self._prep_inputs(m, z, mask)
    
            biases = [mask_bias]
            if z is not None:
                biases.append(z)
        # endregion
        
        if chunk_size is not None:
            m = self._chunk(
                m, 
                biases, 
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )
        else:
            m = self.layer_norm_m(m)
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """
    def __init__(
        self, 
        c_m: int, 
        c_z: int, 
        c_hidden: int, 
        no_heads: int, 
        inf: float = 1e9,
    ):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super().__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.
    """
    def __init__(
        self, 
        c_m: int, 
        c_hidden: int, 
        no_heads: int, 
        inf: float = 1e9
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super().__init__()
        
        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(
        self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        #? 为什么没有use_memory_efficient_kernel, 不知道
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
        """ 
        # [*, N_res, N_seq, C_m]
        m = m.transpose(-2, -3)
        if mask is not None:
            # [*, N_res, N_seq]
            mask = mask.transpose(-1, -2)

        m = self._msa_att(
            m, 
            mask=mask, 
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
        )

        # [*, N_seq, N_res, C_m]
        m = m.transpose(-2, -3)
        #* 这里删除了我认为不必要的代码
        # if mask is not None:
        #     mask = mask.transpose(-1, -2)

        return m


class MSAColumnGlobalAttention(nn.Module):
    """
    Implements Algorithm 19.
    """
    def __init__(
        self, 
        c_in: int, 
        c_hidden: int, 
        no_heads: int, 
        inf: float = 1e9, 
        eps: float = 1e-10,
    ):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
    ) -> torch.Tensor:
        mha_input = {
            "m": m,
            "mask": mask,
        }

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask, use_lma=use_lma)

        return chunk_layer(
            fn,
            mha_input,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, M_res, C_in]
            mask:
                [*, N_seq, N_res]
        """
        if mask is None:
            # [*, N_seq, N_res]
            #* 我修改了这里繁琐的代码，追踪openfold的commit记录发现
            #* 使用注释的代码的原因可能是为了TorchScript
            # mask = torch.ones(
            #     m.shape[:-1],
            #     dtype=m.dtype,
            #     device=m.device,
            # ).detach()
            # requires_grad默认为False
            mask = m.new_ones(m.shape[:-1])

        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size, use_lma=use_lma) 
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask, use_lma=use_lma)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(
        self, 
        c_m: int, 
        n: int,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super().__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"m": m, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m
