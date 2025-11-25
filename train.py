import os
import re
import json
import hydra
import random
import torch
import numpy as np
import lightning as pl
import pandas as pd
from einops import rearrange
from datetime import datetime
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from safetensors.torch import save_file
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from model_utils import adjust_to_ReDirector
from dataset import MultiCamVideoTensorDataset
from lightning.pytorch import seed_everything

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    ):
        super().__init__()

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            redirect_common_files=False,
            model_configs=[ModelConfig(model_id=cfg.model_path, origin_file_pattern="diffusion_pytorch_model*.safetensors", skip_download=True)],
        )

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add modules
        model, module_name_list = adjust_to_ReDirector(getattr(self.pipe, 'dit'), cfg)
        setattr(self.pipe, 'dit', model)

        self.pipe.freeze_except([])

        for name, module in self.pipe.dit.named_modules():
            if any(keyword in name for keyword in module_name_list):
                module.train()
                module.requires_grad_(True)

        # Store other configs
        self.extra_inputs = ['input_latents', 'clip_feature', 'context', 'y', 'traj']

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = cfg.train.max_timestep_boundary
        self.min_timestep_boundary = cfg.train.min_timestep_boundary

        self.learning_rate = cfg.train.lr

    def forward_preprocess(self, data):        
        # CFG-sensitive parameters
        inputs_posi = {}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": data["input_latents"].shape[3] * 8,
            "width": data["input_latents"].shape[4] * 8,
            "num_frames": (data["input_latents"].shape[2] - 1) * 4 + 1,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        
        # Joint distribution modeling; we use the camera trajectory of given video frames as additional input latents
        cond_plucker_embedding = self.pipe.dit.control_adapter.process_camera_coordinates(
            None, inputs_shared["num_frames"], inputs_shared["height"], inputs_shared["width"], None, None, data["traj_cond"]
        )
        cond_plucker_embedding = cond_plucker_embedding[:, :inputs_shared["num_frames"]].permute([0, 4, 1, 2, 3])
        cond_plucker_embedding = torch.concat(
            [
                torch.repeat_interleave(cond_plucker_embedding[:, :, 0:1], repeats=4, dim=2),
                cond_plucker_embedding[:, :, 1:]
            ], dim=2
        )
        cond_plucker_embedding = rearrange(cond_plucker_embedding, 'b c (f k) h w -> b (c k) f h w', k=4)
        cond_plucker_embedding = cond_plucker_embedding.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        inputs_shared['cond_plucker_embedding'] = cond_plucker_embedding
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        return {**inputs_shared, **inputs_posi}

    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device

        with torch.no_grad():
            inputs = self.forward_preprocess(batch)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        self.log("total_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    @rank_zero_only
    def on_train_epoch_end(self):
        checkpoint_dir = os.path.join(self.trainer.default_root_dir, 'checkpoints') 
        os.makedirs(checkpoint_dir, exist_ok=True)

        trainable_param_names = list(
            map(lambda np: np[0].replace('pipe.dit.', ''),
            filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        )
        state_dict = self.pipe.dit.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in trainable_param_names}
        save_file(state_dict, os.path.join(checkpoint_dir, f"step{self.global_step}.safetensors"))

        
@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.1")
def main(cfg: DictConfig):
    rank = int(os.environ.get("RANK", "0"))
    base = 42
    seed = base + rank
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not cfg.wandb.disabled:
        os.environ["WANDB_API_KEY"] = cfg.wandb.wandb_api_key
        wandb_logger = WandbLogger(
            project=cfg.wandb.project_name,
            name = f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg),
        )
    else:
        wandb_logger = False

    model = TrainingModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu",
        devices='auto',
        precision="bf16-mixed",
        strategy='ddp',
        default_root_dir=cfg.output_path,
        log_every_n_steps=1,
        logger=wandb_logger,
        enable_checkpointing=False,
    )

    dataset = MultiCamVideoTensorDataset(cfg.dataset_path, cfg.train.steps_per_epoch * trainer.world_size * cfg.train.batch_size)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader_num_workers
    )

    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()
