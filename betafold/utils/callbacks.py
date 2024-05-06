from typing import Any, Dict, Union
import sys
from tqdm import tqdm

from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

class EarlyStoppingVerbose(EarlyStopping):
    """
        The default EarlyStopping callback's verbose mode is too verbose.
        This class outputs a message only when it's getting ready to stop. 
    """
    def _evalute_stopping_criteria(self, *args, **kwargs):
        should_stop, reason = super()._evalute_stopping_criteria(*args, **kwargs)
        # if should_stop:
        #     rank_zero_info(f"{reason}\n")
        rank_zero_info(f"{reason}\n")

        return should_stop, reason

class TQDMProgressBar4File(TQDMProgressBar):
    """
    针对进度条输出到文件的情形精简了一些输出: 
    1. 删除validation的进度条多余空白行
    2. train batch进度条不会重复刷新
    3. validation时不会刷新train的进度条, 知道validation结束
    4. train batch进度条不会输出v_num
    对于输出到命令行的情形会有问题, 直接使用pl的TQDMProgressBar即可。但建议改写get_metrics如本例所示
    """
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        """
        在tqdm的position中去掉了has_main_bar。原先validation的进度条会在第二行(主进度条会在第一行), 
        在输出到文件时, 会多空白行。
        """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def on_train_batch_end(self, trainer, pl_module, *_: Any) -> None:
        current = self.train_batch_idx + self._val_processed
        if self._should_update(current, self.main_progress_bar.total):
            _update_n(self.main_progress_bar, current, self.get_metrics(trainer, pl_module))
    
    def on_validation_batch_end(self, trainer, *_: Any) -> None:
        if self._should_update(self.val_batch_idx, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, self.val_batch_idx)

        # current = self.train_batch_idx + self._val_processed
        # if trainer.state.fn == "fit" and self._should_update(current, self.main_progress_bar.total):
        #     _update_n(self.main_progress_bar, current)
    
    def get_metrics(self, trainer , pl_module) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
    

def _update_n(bar: tqdm, value: int, ordered_dict = None) -> None:
    if not bar.disable:
        bar.n = value
        if ordered_dict is not None:
            bar.set_postfix(ordered_dict, refresh=False)
        bar.refresh()