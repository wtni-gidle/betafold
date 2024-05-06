import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    DeviceStatsMonitor, 
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from betafold.utils.argparse_utils import remove_arguments
from betafold.utils.callbacks import EarlyStoppingVerbose, TQDMProgressBar4File

        

def main(args):

    # region: callbacks
    callbacks = []

    model_summary = ModelSummary(max_depth=args.max_depth)
    callbacks.append(model_summary)

    tqdm = TQDMProgressBar4File(refresh_rate=1)
    callbacks.append(tqdm)

    #! 突然想到几何势的loss可不可以用带rank的loss
    mc = ModelCheckpoint(
        # dirpath=None, # 默认{logger.log_dir}/checkpoints
        save_last=True, # saves an exact copy of the checkpoint to last.ckpt whenever a checkpoint file gets saved
        filename="Epoch_{epoch:03d}-TrL_{train_loss:.2f}-ValL_{val_loss:.2f}", #todo 待修改, 可能只要log有这些指标就行
        every_n_epochs=1, # 我觉得就每个epoch都check，也不要分step
        save_top_k=-1, 
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True, # 如果False, 在on_validation_end中check, 而不是在之后的on_train_epoch_end
    )
    callbacks.append(mc)

    # openfold的PerformanceLoggingCallback只能算每个batch的时间然后得到一个分布
    # simpleprofiler可以算出每一个重要的hook的耗时，但是必须要在fit结束之后才能将结果写入文件
    #! 既然是profiler，我们就可以尝试让trainer跑10个epoch, max_epoch=10，这样fit就能结束，profiler就有结果。
    #! 然后做好瓶颈分析后，就不需要在trainer中使用profiler了
    if args.profile:
        profiler = SimpleProfiler(
            dirpath=None,
            filename="performance_log",
            extended=True,
        )
        callbacks.append(profiler)
        
        device_monitor = DeviceStatsMonitor()
        callbacks.append(device_monitor)

    if args.early_stopping:
        es = EarlyStoppingVerbose(
            monitor="val/lddt_ca", #todo, 待修改
            mode="max", #todo, 待修改
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False, # 使用修改过的es，这里就要设置为False
            check_finite=True,
        )
        callbacks.append(es)

    if args.log_lr:
        # 会作用在lightningmodule.log()中
        lr_monitor = LearningRateMonitor()
        callbacks.append(lr_monitor)
    # endregion

    # region: loggers, 只考虑使用TensorBoardLogger或WandbLogger
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "tb_logs"),
        name=args.exp_name, # "experiment"实际上是一组相关的运行。你可能会运行多个实验，每个实验都有不同的超参数设置或模型架构。
        version=args.exp_version, # 每个实验可能包含一个或多个运行，每个运行代表一次模型训练过程的记录。
        max_queue=10, # 指定了在内存中保存日志的数量，即内存队列的最大长度。超过这个值就会写入磁盘
        flush_secs=120, # 超过刷新时间间隔就会写入磁盘。符合这两个条件之一就会将记录写入磁盘。值越大，内存需求越大，但io越少，训练速度越快
    )
    loggers.append(tb_logger)
    # endregion

    if args.deepspeed_config_path is not None:
        strategy = DeepSpeedStrategy(config=args.deepspeed_config_path) #! PL警告：DeepSpeedStrategy处于测试阶段
    elif (args.devices is not None and args.devices > 1) or args.num_nodes > 1:
        # 在调试阶段，find_unused_parameters设置为True；如果模型已经完备了，设置为False可以提高性能
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = None


    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        ### parameters maybe used
        # --------------------------------------------------
        # accumulate_grad_batches=None, # 梯度累加, not used here
        # auto_scale_batch_size=False, # 自动寻找batch_size, 对于drfold的小模型，可能会用到
        # check_val_every_n_epoch=1, # 每n个epoch执行一次valiation。
        # val_check_interval=None, # 这两个参数共同影响了模型训练过程中验证集评估的频率，可以根据具体需求来调整
        # deterministic=False, # 不确定要不要设置这个参数，openfold没有设置
        # gradient_clip_val=None, # 梯度裁剪, not used here
        # limit_train_batches=1.0, # debug时候用的
        # log_every_n_steps=50, # log的频率
        # profiler=None, # 分析代码性能
        # sync_batchnorm=False, #有batchnorm层才会用到
        # reload_dataloaders_every_n_epochs=1, # This re-applies the training-time filters at the beginning of every epoch
        # devices=None, # 每个node的gpu个数
        # num_nodes=1,
        # precision=32,
        # --------------------------------------------------
        default_root_dir=args.output_dir, 
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser = pl.Trainer.add_argparse_args(parser)

    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )
    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser, 
        [
            "--accelerator", 
            "--gpus", 
            "--num_processes", 
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_n_epochs",
        ]
    )
    args = parser.parse_args()

    # region: check args
    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1
    # endregion
