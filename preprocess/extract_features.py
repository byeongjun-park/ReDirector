import torch, os
import lightning as pl
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from omegaconf import DictConfig
import hydra
from dataset import TextVideoDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ExtractionModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            redirect_common_files=False,
            model_configs=[
                ModelConfig(model_id=cfg.model_path, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", skip_download=True),
                ModelConfig(model_id=cfg.model_path, origin_file_pattern="Wan2.1_VAE.pth", skip_download=True),
                ModelConfig(model_id=cfg.model_path, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", skip_download=True),
            ],
            tokenizer_config = ModelConfig(model_id=cfg.model_path, origin_file_pattern="google/*", skip_download=True),
        )
        self.reverse = cfg.extract.reverse
        self.pipe.load_models_to_device(["text_encoder","image_encoder","vae"])

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def forward_preprocess(self, data):        
        if data['v_cond'] is not None:
            prompt_emb = self.pipe.prompter.encode_prompt(data["prompt"], positive=None, device=self.pipe.device)
            clip_context = self.pipe.image_encoder.encode_image([data['v_cond'][:, :, 0]]).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            y = self.pipe.vae.encode(data["v_cond"], device=self.pipe.device, tiled=False)

            # save tensor
            for i, path in enumerate(data['path']):
                save_path = path + ".reversed_landscape_tensors.pth" if self.reverse else path + ".landscape_tensors.pth"
                data = {"context": prompt_emb[i], "clip_feature": clip_context[i], "y": y[i]}
                torch.save(data, save_path)
        else:
            print ('Wrong data loaded')

    def test_step(self, batch, batch_idx):
        self.pipe.device = self.device

        with torch.no_grad():
            inputs = self.forward_preprocess(batch)

        return inputs


@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.1")
def main(cfg: DictConfig):
    dataset = TextVideoDataset(
        cfg.dataset_path,
        max_num_frames=cfg.extract.num_frames,
        frame_interval=1,
        num_frames=cfg.extract.num_frames,
        reverse=cfg.extract.reverse,
        height=cfg.extract.height,
        width=cfg.extract.width,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.extract.batch_size,
        num_workers=cfg.dataloader_num_workers
    )

    model = ExtractionModule(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices='auto',
        precision="bf16",
        default_root_dir=cfg.output_path,
        logger=None
    )
    trainer.test(model, dataloader)


if __name__ == '__main__':
    main()
