import os
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from lightning.pytorch.callbacks import Callback
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
import uuid
import time

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

class IterationTimer(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module._step_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - pl_module._step_start_time
        pl_module.log("time/step_time", elapsed, prog_bar=True)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    print(cfg)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))


    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        if cfg.checkpointing.load is not None and cfg.checkpointing.resume:
            print(cfg.checkpointing.load, Path(cfg.checkpointing.load).parent.parent)
            with open( Path(cfg.checkpointing.load).parent.parent / "wandb_run_id.txt") as f:
                resume_id = f.read().strip()
            logger = WandbLogger(
                project=cfg_dict.wandb.project,
                mode=cfg_dict.wandb.mode,
                name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
                tags=cfg_dict.wandb.get("tags", None),
                log_model=False,
                save_dir=output_dir,
                config=OmegaConf.to_container(cfg_dict),
                id=resume_id,  
                resume="allow",
            )

            # save resume_id
            with open(output_dir / "wandb_run_id.txt", "w") as f:
                f.write(str(resume_id))

        else:
            new_id = uuid.uuid4().hex
            logger = WandbLogger(
                project=cfg_dict.wandb.project,
                mode=cfg_dict.wandb.mode,
                name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
                tags=cfg_dict.wandb.get("tags", None),
                log_model=False,
                save_dir=output_dir,
                config=OmegaConf.to_container(cfg_dict),
                id=new_id,  
            )
            # save new_id
            with open(output_dir / "wandb_run_id.txt", "w") as f:
                f.write(str(new_id))
        
        callbacks.append(LearningRateMonitor("step", True))
        callbacks.append(IterationTimer())

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            # print("load", ckpt_weights.keys())
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
            # print("missing_keys", missing_keys)
            # print("unexpected_keys", unexpected_keys)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")
        
    model_kwargs = {
        'optimizer_cfg': cfg.optimizer,
        'test_cfg': cfg.test,
        'train_cfg': cfg.train,
        'encoder': encoder,
        'encoder_visualizer': encoder_visualizer,
        'decoder': get_decoder(cfg.model.decoder),
        'losses': get_losses(cfg.loss),
        'step_tracker': step_tracker,
        'distiller': distiller,
    }

    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights but no optimizer state
        print(f"Loading full model weights from {checkpoint_path}")
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path,
            **model_kwargs,
            strict=True,
            map_location="cpu",
        )
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    
    
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path  if cfg.checkpointing.resume else None,)
    else:
        model_wrapper.ckpt_path = checkpoint_path
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
