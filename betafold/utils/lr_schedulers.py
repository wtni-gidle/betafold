import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class AlphaFoldLRScheduler(_LRScheduler):
    """ Implements the learning rate schedule defined in the AlphaFold 2
        supplement. A linear warmup is followed by a plateau at the maximum
        learning rate and then exponential decay.
         
        Note that the initial learning rate of the optimizer in question is 
        ignored; use this class' base_lr parameter to specify the starting 
        point of the warmup.
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        last_epoch: int = -1, 
        base_lr: float = 0.,
        max_lr: float = 0.001,
        warmup_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ):
        step_counts = {
            "warmup_steps": warmup_steps,
            "start_decay_after_n_steps": start_decay_after_n_steps,
        }

        for k, v in step_counts.items():
            if v < 0:
                raise ValueError(f"{k} must be nonnegative")

        if warmup_steps > start_decay_after_n_steps:
            raise ValueError(
                "warmup_steps must not exceed start_decay_after_n_steps"
            )

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr
            self.base_lrs.append(self.base_lr)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )

        step_no = self.last_epoch

        if step_no <= self.warmup_steps:
            return [(self.max_lr - base_lr) * step_no / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        elif step_no > self.start_decay_after_n_steps:
            steps_since_decay = step_no - self.start_decay_after_n_steps
            exp = (steps_since_decay // self.decay_every_n_steps) + 1
            return [self.max_lr * (self.decay_factor ** exp) for _ in self.base_lrs]
        else: # plateau
            return [self.max_lr for _ in self.base_lrs]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine Annealing with Warmup Restarts scheduler.

    This scheduler implements the cosine annealing with warmup restarts learning rate schedule,
    which combines the cosine annealing schedule with warmup and restarts strategies.

    https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): Number of steps in the first cycle.
        cycle_mult(float): Multiplicative factor by which the number of steps in each cycle will change.
            Default is 1.0, indicating no change.
        max_lr(float): Maximum learning rate for cosine annealing in the first cycle.
            Default is 0.1.
        min_lr(float): Minimum learning rate for cosine annealing.
            Default is 0.001.
        warmup_steps(int): Number of warmup steps.
            Default is 0.
        gamma(float): Multiplicative factor by which to decay the maximum learning rate after each cycle. 
            Default is 1.0, indicating no decay.
        last_epoch (int): The index of the last epoch. Default is -1.
    Note:
        - This scheduler implements a learning rate schedule that combines warmup, cosine annealing,
          and cycle restarts strategies.
        - The warmup period gradually increases the learning rate from `min_lr` to `max_lr`.
        - The learning rate then follows the cosine annealing schedule until the end of the cycle,
          at which point the cycle restarts.
        - After each cycle restart, the maximum learning rate is decayed by a factor of `gamma`.
    """
    def __init__(
        self,
        optimizer : torch.optim.Optimizer,
        first_cycle_steps : int,
        cycle_mult : float = 1.,
        max_lr : float = 0.1,
        min_lr : float = 0.001,
        warmup_steps : int = 0,
        gamma : float = 1.,
        last_epoch : int = -1
    ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        """
        这个函数覆盖了父类中对`base_lrs`的定义, 并且废弃`param_group['initial_lr']`的使用. 
        修改`param_group['lr']`
        """
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = self.min_lr
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
        
        if self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        # region: 计算last_epoch, step_in_cycle, cycle, cur_cycle_steps
        if epoch is None:
            self.last_epoch += 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            self.last_epoch = epoch
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        # endregion        
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnnealingWarmupRestarts2(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts2, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr