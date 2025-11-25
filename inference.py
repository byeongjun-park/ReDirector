import torch
from diffsynth.models import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import imageio
from torchvision.transforms import v2
from einops import rearrange
import torchvision
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from omegaconf import DictConfig
import hydra
import os
from model_utils import adjust_to_ReDirector, save_video, make_camera_trajectory
import subprocess
import shutil, glob
from natsort import natsorted
import imageio.v3 as iio
from pathlib import Path

def load_video_with_captions(cfg, save_dir):
    frame_process = v2.Compose([
        v2.CenterCrop(size=(cfg.eval.height, cfg.eval.width)),
        v2.Resize(size=(cfg.eval.height, cfg.eval.width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    caption_processor = AutoProcessor.from_pretrained(cfg.blip_path)
    captioner = Blip2ForConditionalGeneration.from_pretrained(cfg.blip_path, torch_dtype=torch.float16).to("cuda")

    if cfg.eval.video_path.endswith('.mp4'):
        reader = imageio.get_reader(cfg.eval.video_path)
        num_frames = reader.count_frames()

        if num_frames % 4 != 0:
            eval_frames = (num_frames // 4) * 4 + 1
        else:
            eval_frames = num_frames - 3

        fps = reader.get_meta_data()['fps']

        frames = []
        for frame_id in range(eval_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)

            _width, _height = frame.size
            scale = max(cfg.eval.width / _width, cfg.eval.height / _height)
            frame = torchvision.transforms.functional.resize(
                frame,
                (round(_height*scale), round(_width*scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

    elif os.path.isdir(cfg.eval.video_path):
        files = natsorted(
            glob.glob(os.path.join(cfg.eval.video_path, "*.png")) +
            glob.glob(os.path.join(cfg.eval.video_path, "*.jpg")) +
            glob.glob(os.path.join(cfg.eval.video_path, "*.jpeg"))
        )

        num_frames = len(files)

        if num_frames % 4 != 0:
            eval_frames = (num_frames // 4) * 4 + 1
        else:
            eval_frames = num_frames - 3

        fps = 30  # default fps
        frames = []
        for f in files[:eval_frames]:
            frame = Image.fromarray(iio.imread(f)) 

            _width, _height = frame.size
            scale = max(cfg.eval.width / _width, cfg.eval.height / _height)
            frame = torchvision.transforms.functional.resize(
                frame,
                (round(_height*scale), round(_width*scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            frame = frame_process(frame)
            frames.append(frame)

    frames = torch.stack(frames, dim=0)

    # extract intrinsic/extrinsic camera parameters using ViPE
    video_dir = Path(save_dir).parents[0]
    if not os.path.exists(os.path.join(video_dir, 'input.mp4')):
        save_video(
            (frames.permute(0, 2, 3, 1) + 1) / 2,
            os.path.join(video_dir, 'input.mp4'),
            fps=fps,
        )
        subprocess.run(f'vipe infer {os.path.join(video_dir, "input.mp4")} --output {os.path.join(video_dir, "input")} --pipeline only_pose', shell=True, check=True)

    # extract text caption
    video = rearrange(frames, "T C H W -> C T H W")
    image_array = ((video[:, video.shape[1]//2] + 1) * 127.5).cpu().numpy().astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    pil_image = Image.fromarray(image_array)
    inputs = caption_processor(images=pil_image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = captioner.generate(**inputs)
    generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # calcuate relative poses
    intrinsic = np.load(os.path.join(video_dir, "input", "intrinsics", "input.npz"))['data']
    fx = intrinsic[0, 0] / cfg.eval.width
    fy = intrinsic[0, 1] / cfg.eval.height
    intrinsics = torch.tensor([0, fx, fy, 0.5, 0.5, 0, 0]).unsqueeze(0).repeat(eval_frames, 1)

    cond_pose = np.load(os.path.join(video_dir, "input", "pose", "input.npz"))['data']  # c2w
    cond_c2ws = np.linalg.inv(cond_pose[0])[None] @ cond_pose
    cond_cam_data = rearrange(torch.as_tensor(cond_c2ws[:, :3], dtype=intrinsics.dtype), 'b c d -> b (c d)')
    cond_cam_data = torch.cat([intrinsics, cond_cam_data], dim=1)

    # load camera
    target_c2ws = make_camera_trajectory(cfg, eval_frames)
    target_cam_data = rearrange(torch.as_tensor(target_c2ws[:, :3], dtype=intrinsics.dtype), 'b c d -> b (c d)')
    target_cam_data = torch.cat([intrinsics, target_cam_data], dim=1)

    data = {
        "text": generated_text,
        "cond_video": video.unsqueeze(0), 
        'traj': target_cam_data.unsqueeze(0),
        'traj_cond': cond_cam_data.unsqueeze(0)
    }

    return data, eval_frames, fps

@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.1")
def main(cfg: DictConfig):
    video_name = os.path.splitext(os.path.basename(os.path.normpath(cfg.eval.video_path)))[0]
    os.makedirs(cfg.eval.output_dir, exist_ok=True)

    data, num_frames, fps = load_video_with_captions(cfg, cfg.eval.output_dir)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        redirect_common_files=False,
        model_configs=[
            ModelConfig(model_id=cfg.model_path, origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
            ModelConfig(model_id=cfg.model_path, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
            ModelConfig(model_id=cfg.model_path, origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
            ModelConfig(model_id=cfg.model_path, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu", skip_download=True),
        ],
        tokenizer_config = ModelConfig(model_id=cfg.model_path, origin_file_pattern="google/*", skip_download=True),
    )

    setattr(pipe, 'dit', adjust_to_ReDirector(getattr(pipe, 'dit'), cfg)[0])

    state_dict = load_state_dict(cfg.eval.ckpt_path, torch_dtype=pipe.torch_dtype, device=pipe.device)
    _, unexpected_keys = pipe.dit.load_state_dict(state_dict, strict=False)

    assert len(unexpected_keys) == 0

    pipe.enable_vram_management()

    video = pipe(
        prompt=data['text'],
        negative_prompt=cfg.eval.negative_prompt,
        first_frame=data['cond_video'][:, :, 0],
        cond_video=data['cond_video'],
        traj=data['traj'].to('cuda'),
        traj_cond=data['traj_cond'].to('cuda'),
        height=cfg.eval.height, width=cfg.eval.width, num_frames=num_frames,
        cfg_scale=cfg.eval.cfg_scale,
        seed=0, tiled=False
    )

    save_video(video, f"{cfg.eval.output_dir}/{video_name}_camera_{cfg.eval.cam_type}_speed_{cfg.eval.cam_speed}.mp4", fps=fps)

if __name__ == '__main__':
    main()
 

