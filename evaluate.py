import torch
from featup.util import norm
import numpy as np
import os
from einops import rearrange
from vipe.vipe.utils.io import read_depth_artifacts, read_rgb_artifacts, read_instance_artifacts
import cv2
# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

from met3r import MEt3R
import shutil
import subprocess
import argparse
from model_utils import make_camera_trajectory
from omegaconf import OmegaConf
from pathlib import Path
from eval_utils import eval_rot, eval_trans

def resize_video(input_path, output_path, size=(224, 224)):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, size)
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"Saved resized video to {output_path}")

# generated
def get_vipe_info(vipe_dir, name):
    intrinsic = np.load(os.path.join(vipe_dir, name, "intrinsics", f"{name}.npz"))['data']
    extrinsic = np.load(os.path.join(vipe_dir, name, "pose", f"{name}.npz"))['data']
    
    intrinsic = torch.as_tensor(intrinsic)
    extrinsic = torch.as_tensor(extrinsic)
    extrinsic = torch.linalg.inv(extrinsic[:1]) @ extrinsic # set the first frame as reference
    depth = torch.stack([d for _, d in read_depth_artifacts(os.path.join(vipe_dir, name, "depth", f"{name}.zip"))])
    mask = torch.stack([d for _, d in read_instance_artifacts(os.path.join(vipe_dir, name, "mask", f"{name}.zip"))])
    rgb = torch.stack([c for _, c in read_rgb_artifacts(os.path.join(vipe_dir, name, "rgb", f"{name}.mp4"))])

    B, H, W = depth.shape
    fx, fy, cx, cy = intrinsic.unbind(dim=-1)

    # pixel grid (u,v)
    v, u = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing="ij"
    )
    u = u.float()[None].expand(B, -1, -1)
    v = v.float()[None].expand(B, -1, -1)

    Z = depth
    X = (u - cx[:, None, None]) * Z / fx[:, None, None]
    Y = (v - cy[:, None, None]) * Z / fy[:, None, None]

   # camera coords (homogeneous)
    pts_cam = torch.stack((X, Y, Z, torch.ones_like(Z)), dim=-1)  # (B, H, W, 4)

    # reshape for matmul
    pts_cam_flat = pts_cam.view(B, -1, 4).transpose(1, 2)  # (B, 4, H*W)

    # transform to world coords
    pts_world = extrinsic @ pts_cam_flat  # (B, 4, H*W)

    # back to (B, H, W, 3)
    pts_world = pts_world.transpose(1, 2)[..., :3].view(B, H, W, 3)

    return {
        'intrinsic': intrinsic,
        'extrinsic': extrinsic,
        'pts': pts_world,
        'mask': (mask != 0).float(),
        'rgb': rgb,
        'size': (B, H, W)
    }

##############################################################################################################
@torch.no_grad()
def calcuate_dyn_met3r(dir_name, video_name, size=224):
    resized_pred_name = 'resized_' + video_name
    resize_video(os.path.join(dir_name, f'{video_name}.mp4'), os.path.join(dir_name, f'{resized_pred_name}.mp4'), size=(size, size))
    subprocess.run(f'vipe infer {os.path.join(dir_name, f"{resized_pred_name}.mp4")} --output {os.path.join(dir_name, resized_pred_name)}', shell=True, check=True)

    pred_info = get_vipe_info(dir_name, resized_pred_name)
    B, H, W = pred_info['size']

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.005, 
        points_per_pixel=10,
        bin_size=0,
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=None, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    upsampler = torch.hub.load('mhamilton723/FeatUp', 'dino16', use_norm=True).eval()
    upsampler = upsampler.to('cuda')

    src_indices = np.linspace(0, B - 1, B // 2 + 1, dtype=np.int64)
    tgt_indcies = np.delete(np.arange(B), src_indices)

    fx, fy, cx, cy = pred_info['intrinsic'].chunk(4, dim=-1)
    camera_matrix = torch.stack([torch.tensor([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1]]) for _fx, _fy, _cx, _cy in zip(fx.squeeze(), fy.squeeze(), cx.squeeze(), cy.squeeze())]).to('cuda')
    viewmat = torch.linalg.inv(pred_info['extrinsic']).to('cuda')
    pts = pred_info['pts'].to('cuda')
    mask = pred_info['mask'].bool().to('cuda')  # zero is the background
    
    feat_list = []
    rgb = rearrange(pred_info['rgb'], 'b h w c -> b c h w').to('cuda')
    with torch.amp.autocast('cuda', enabled=True):
        for tmp in rgb:
            feat = upsampler(norm(tmp[None]))
            feat_list.append(feat)
    feats = torch.cat(feat_list, dim=0)

    image_size = torch.tensor([[H, W]]).repeat(len(tgt_indcies), 1)
    
    cameras = cameras_from_opencv_projection(
                R=viewmat[tgt_indcies, :3, :3],
                tvec=viewmat[tgt_indcies, :3, -1],
                camera_matrix=camera_matrix[tgt_indcies],
                image_size=image_size
            )   

    bg_feat = [-10000] * feats.shape[1]
    src_pts = rearrange(pts[src_indices], 'b h w c -> (b h w) c')
    src_feat = rearrange(feats[src_indices], 'b c h w -> (b h w) c')
    src_masks = rearrange(mask[src_indices], 'b h w -> (b h w)')

    score_list = []
    for i in range(len(cameras)):
        current_pc = Pointclouds(points=[src_pts[~src_masks]], features=[src_feat[~src_masks]])
        rendered_feat = renderer(current_pc, cameras=cameras[i], background_color=bg_feat)
        rendered_feat = rearrange(rendered_feat, 'b h w c -> (b h w) c')
        gt_feat = rearrange(feats[tgt_indcies[i]], 'c h w -> (h w) c')
        mask_i = rearrange(mask[tgt_indcies[i]], 'h w -> (h w)')
        score = torch.nn.functional.cosine_similarity(gt_feat[~mask_i], rendered_feat[~mask_i], dim=-1)
        score_list.append(score)
    scores = torch.cat(score_list, dim=0).mean()

    del upsampler, renderer
    torch.cuda.empty_cache()

    return round(scores.item(), 4)

##############################################################################################################

# Initialize MEt3R
def calcuate_MEt3R(dir_name, video_name):
    metric = MEt3R(
        img_size=None, # Default to 256, set to `None` to use the input resolution on the fly!
        use_norm=True, # Default to True 
        backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
        feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
        feature_backbone_weights="mhamilton723/FeatUp", # Default
        upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
        distance="cosine", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
        freeze=True, # Default to True
    ).cuda()

    # Prepare inputs of shape (batch, views, channels, height, width): views must be 2
    # RGB range must be in [-1, 1]
    # Reduce the batch size in case of CUDA OOM
    input_rgb = torch.stack([c for _, c in read_rgb_artifacts(os.path.join(dir_name, "input.mp4"))])
    pred_rgb = torch.stack([c for _, c in read_rgb_artifacts(os.path.join(dir_name, f"{video_name}.mp4"))])

    if (input_rgb.shape[1] != pred_rgb.shape[1]) or (input_rgb.shape[2] != pred_rgb.shape[2]):
        pred_rgb = torch.nn.functional.interpolate(rearrange(pred_rgb, 'b h w c -> b c h w'), size=(input_rgb.shape[1], input_rgb.shape[2]), mode='bilinear', align_corners=False)
        pred_rgb = rearrange(pred_rgb, 'b c h w -> b h w c')

    if input_rgb.shape[0] != pred_rgb.shape[0]:
        pred_rgb = pred_rgb[:input_rgb.shape[0]]

    input_rgb = rearrange(input_rgb, 'b h w c -> b () c h w')
    pred_rgb = rearrange(pred_rgb, 'b h w c -> b () c h w')

    inputs = torch.cat([input_rgb, pred_rgb], dim=1).cuda()
    inputs = 2 * inputs - 1
    inputs = inputs.clip(-1, 1)

    # Evaluate MEt3R
    score_list = []
    for i in range(inputs.shape[0]):  # due to memeory issue
        score, *_ = metric(
            images=inputs[i].unsqueeze(0), 
            return_overlap_mask=False, # Default 
            return_score_map=False, # Default 
            return_projections=False # Default 
        )

        score_list.append(score)

    scores  = torch.stack(score_list).mean().item()
    return round(scores, 4)

def rotation_error(pred_extrinsic: torch.Tensor, gt_extrinsic: torch.Tensor, in_degrees=True):
    """
    pred_extrinsic, gt_extrinsic: (N, 4, 4) torch.Tensor
    Return: mean rotation error (scalar)
    """
    assert pred_extrinsic.shape == gt_extrinsic.shape
    N = pred_extrinsic.shape[0]

    R_pred = pred_extrinsic[:, :3, :3]
    R_gt   = gt_extrinsic[:, :3, :3]

    # Relative rotation
    R_rel = torch.matmul(R_pred, R_gt.transpose(1,2))  # (N,3,3)

    # trace -> angle
    trace = R_rel[:, 0,0] + R_rel[:,1,1] + R_rel[:,2,2]
    cos_theta = (trace - 1) / 2
    # clamp numerical issues
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (N,)

    return theta.sum()

def translation_error(pred_extrinsic: torch.Tensor, gt_extrinsic: torch.Tensor):
    """
    pred_extrinsic, gt_extrinsic: (N, 4, 4) torch.Tensor
    Return: mean translation error after scale alignment
    """
    assert pred_extrinsic.shape == gt_extrinsic.shape
    N = pred_extrinsic.shape[0]

    T0_pred_inv = torch.linalg.inv(pred_extrinsic[0])
    T0_gt_inv   = torch.linalg.inv(gt_extrinsic[0])
    T_pred_rel = torch.matmul(T0_pred_inv[None], pred_extrinsic)   # (N,4,4)
    T_gt_rel   = torch.matmul(T0_gt_inv[None], gt_extrinsic)

    t_pred = T_pred_rel[:, :3, 3]
    t_gt   = T_gt_rel[:, :3, 3]

    dist_pred = torch.norm(t_pred[-1] - t_pred[0])
    dist_gt   = torch.norm(t_gt[-1] - t_gt[0])

    scale = dist_gt / (dist_pred + 1e-8)

    t_pred_scaled = t_pred * scale

    # per-frame L2 error
    errs = torch.norm(t_pred_scaled - t_gt, dim=1)  # (N,)

    return errs.sum()  # mean error, per-frame errors

def calculate_pose_error(dir_name, video_name, cfg):
    subprocess.run(f'vipe infer {os.path.join(dir_name, f"{video_name}.mp4")} --output {os.path.join(dir_name, video_name)} --pipeline only_pose', shell=True, check=True)
    pred_c2ws = np.load(os.path.join(dir_name, video_name, "pose", f"{video_name}.npz"))['data']
    pred_c2ws = torch.as_tensor(pred_c2ws, dtype=torch.float32)

    # CamCtrl
    cond_c2ws = make_camera_trajectory(cfg, eval_frames=pred_c2ws.shape[0])  # (N,4,4)
    cond_c2ws = torch.as_tensor(cond_c2ws, dtype=torch.float32)
    
    RotErr = eval_rot(cond_c2ws[:, :3, :3], pred_c2ws[:, :3, :3])
    TransErr = eval_trans(cond_c2ws[:, :3, 3], pred_c2ws[:, :3, 3])

    return round(RotErr, 2), round(TransErr, 2)  # in degrees

def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--data_path",
        type=str,
        default='results/bear_camera_1_speed_1.mp4',
    )
    args = parser.parse_args()
    return args

##############################################################################################################

if __name__ == "__main__":
    # configs
    args = parse_args()
    dir_name = Path(args.data_path).parents[0]
    video_name = os.path.splitext(os.path.basename(os.path.normpath(args.data_path)))[0]
    infor = video_name.split('_')
    cam_type, cam_speed = infor[-3], infor[-1]
    cfg = OmegaConf.create({
        "eval": {
            "camera_path": "dataset/camera_extrinsics.json",
            "cam_type": int(cam_type),
            "cam_speed": float(cam_speed),
        }
    })

    assert cfg.eval.cam_type > 0
    output_path = Path(dir_name) / f'results.txt'

    try:
        RotErr, TransErr = calculate_pose_error(dir_name, video_name, cfg)
    except Exception as e:
        RotErr, TransErr = None, None

    try:
        dyn_met3r = calcuate_dyn_met3r(dir_name, video_name)
    except Exception as e:
        dyn_met3r = None

    try:
        pairwise_met3r = calcuate_MEt3R(dir_name, video_name)
    except Exception as e:
        pairwise_met3r = None

    with output_path.open('w', encoding='utf-8') as f:
        f.write(f'TransErr: {TransErr}\n')
        f.write(f'RotErr: {RotErr}\n')
        f.write(f'Dyn-MEt3R: {dyn_met3r}\n')
        f.write(f'MEt3R: {pairwise_met3r}\n')

    shutil.rmtree(os.path.join(dir_name, video_name))  # remove ViPE folder
    shutil.rmtree(os.path.join(dir_name, 'resized_' + video_name))  # remove tmp folder
    os.remove(os.path.join(dir_name, f'resized_{video_name}.mp4'))  # remove tmp video
    