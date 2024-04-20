"""
ema是一种给予近期数据更高权重的平均方法。
当decay较大时, EMA对新观测值的适应性会变慢, 更多地依赖于历史观测值, 因此平均值的变化会相对稳定。
较小的decay值可以加快训练启动速度, 更激进。
在深度学习的优化过程中, 会将模型参数拷贝出来做ema(称为影子权重), 而不是就地更新。
影子权重不会参与训练, 每次模型权重经过梯度下降更新后, 就会应用ema计算影子权重。
影子权重应该是用于evaluation, 具体用处不详
"""
from typing import Dict
from collections import OrderedDict
import torch
import torch.nn as nn

from betafold.utils.tensor_utils import tensor_tree_map


class ExponentialMovingAverage:
    """
    Maintains moving averages of parameters with exponential decay

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`

    where `decay` is an attribute of the ExponentialMovingAverage object.
    """
    def __init__(self, model: nn.Module, decay: float):
        """
        Args:
            model:
                A torch.nn.Module whose parameters are to be tracked
            decay:
                A value (usually close to 1.) by which updates are
                weighted as part of the above formula
        """
        super().__init__()

        clone_param = lambda t: t.clone().detach()
        #* model.state_dict()除了包含可学习的参数parameters, 还包含不参与反向传播的参数buffer
        self.params: Dict[str, torch.Tensor] = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device

    def to(self, device: torch.device) -> None:
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict_(self, update: Dict, state_dict: Dict):
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        """
        Updates the stored parameters using the state dict of the provided
        module. The module should have the same structure as that used to
        initialize the ExponentialMovingAverage object.
        """
        self._update_state_dict_(model.state_dict(), self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        for k in state_dict["params"].keys():
            self.params[k] = state_dict["params"][k].clone()
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )
