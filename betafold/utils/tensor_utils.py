from functools import partial
from typing import List, Callable, Any, Dict

import torch
import torch.nn as nn


def add(m1: torch.Tensor, m2: torch.Tensor, inplace: bool):
    """是否原地操作, 这个在checkpoint中被禁用, 可用于inference"""
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def add_first_dims(tensor: torch.Tensor, no_dims: int) -> torch.Tensor:
    """
    在张量前添加指定数量的维度。

    Args:
        tensor (torch.Tensor): 输入张量。
        no_dims (int): 要添加的维度数量。

    Returns:
        torch.Tensor: 添加维度后的张量。

    Example:
        如果张量的形状为(2, 3), no_dims = 2, 
        则在张量前添加两个维度后，形状变为(1, 1, 2, 3)。
    """
    new_shape = (1,) * no_dims + tensor.shape
    return tensor.reshape(new_shape)


def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    """
    Permute the final dimensions of a PyTorch tensor based on given indices.

    Args:
        tensor (torch.Tensor): The input tensor.
        inds (List[int]): A list of indices specifying the final dimensions' order.

    Returns:
        torch.Tensor: Permuted tensor.

    Example:
        If tensor has shape [A, B, C, D] and inds = [1, 2, 0],
        then the resulting tensor is of shape [A, C, D, B].
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    """
    将张量的最后几个维度展平为一个维度。

    Args:
        t (torch.Tensor): 输入张量。
        no_dims (int): 要保留的维度数量。

    Returns:
        torch.Tensor: 展平后的张量。

    Example:
        如果张量的形状为[A, B, C, D], no_dims = 2, 
        则最后两个维度将被展平为一个维度，得到形状为[A, B, C * D]的张量。
    """
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(
    mask: torch.Tensor, 
    value: torch.Tensor, 
    dim: int, 
    eps: float = 1e-4
) -> torch.Tensor:
    """
    计算在给定维度上的加权平均值，其中权重由掩码张量指定。

    Args:
        mask (torch.Tensor): 用于指定权重的二进制掩码张量。
        value (torch.Tensor): 包含数值的张量，将根据掩码进行加权平均。
        dim (int): 执行加权平均的维度。
        eps (float, optional): 一个小的常数, 用于防止除以零。默认为1e-4。

    Returns:
        torch.Tensor: 在给定维度上的加权平均值。

    Example:
        如果 mask 是一个形状为 (2, 3) 的二进制张量, value 是一个形状为 (2, 3, 4) 的张量，
        则可以使用 masked_mean(mask, value, dim=2) 来计算在第三个维度上的加权平均值。
    """
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def dict_multimap(fn: Callable[[List], Any], dicts: List[Dict]) -> Dict:
    """
    对输入的字典列表中的每个字典进行映射操作，并将结果组合成一个新的字典返回。

    Args:
        fn (function): 用于映射操作的函数。
        dicts (list): 包含字典的列表。

    Returns:
        dict: 映射结果组成的新字典。

    Example:
        如果有以下字典列表：
        dicts = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        则可以使用 dict_multimap(lambda x: x*2, dicts) 对每个值进行乘以2的映射操作。
        结果将是 {'a': [2, 6], 'b': [4, 8]}。
    """
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    """
    Algorithm 5 One-hot encoding with nearest bin
    将输入张量编码为 one-hot 形式。

    Args:
        x (torch.Tensor): 输入张量。[*]
        v_bins (torch.Tensor): 包含编码值的张量。[num_classes]

    Returns:
        torch.Tensor: one-hot 编码的张量。[*, num_classes]

    Example:
        如果 x 是一个形状为 (2, 3) 的张量, v_bins 是一个形状为 (4,) 的张量，
        则可以使用 one_hot(x, v_bins) 将 x 编码为 one-hot 形式。
    """
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    # [*]
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()

# * tough
def batched_gather(
    data: torch.Tensor, 
    inds: torch.Tensor, 
    dim: int = 0, 
    no_batch_dims: int = 0
) -> torch.Tensor:
    """
    对张量进行批次选择操作。
    这玩意看不懂, 到具体用的时候再具体分析

    Args:
        data (torch.Tensor): 输入张量。
        inds (torch.Tensor): 包含索引的张量，用于选择元素。
        dim (int, optional): 执行选择操作的维度。默认为0。
        no_batch_dims (int, optional): 批次维度的数量。默认为0。

    Returns:
        torch.Tensor: 选择后的张量。

    Example:
        如果 data 是一个形状为 (2, 3, 4) 的张量, inds 是一个形状为 (2, 2) 的张量，
        则可以使用 batched_gather(data, inds, dim=1) 对张量进行批次选择操作。
    """
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


# With tree_map, a poor man's JAX tree_map
def dict_map(fn: Callable, dic: Dict, leaf_type) -> Dict:
    """
    对嵌套字典结构中的每个叶子节点应用给定的函数。

    Args:
        fn (Callable): 要应用的函数。
        dic (Dict): 嵌套字典结构。
        leaf_type: 叶子节点的类型。

    Returns:
        Dict: 新的字典结构，其中叶子节点被替换为应用函数后的值。
    """
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn: Callable, tree: Any, leaf_type: Any):
    """
    对任意嵌套结构中的每个叶子节点应用给定的函数。

    Args:
        fn (Callable): 要应用的函数。
        tree: 嵌套结构（字典、列表或元组）。
        leaf_type: 叶子节点的类型。

    Returns:
        object: 新的嵌套结构，其中叶子节点被替换为应用函数后的值。
    """
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")
    

tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
