import torch
import random
import numpy as np
import os
import re
import json
import pandas as pd
from einops import rearrange
import imageio
import torchvision
from torchvision.transforms import v2
from PIL import Image

class MultiCamVideoTensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, steps_per_epoch):
        metadata = pd.read_csv('dataset/metadata.csv')
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.original_path = [i + ".landscape_tensors.pth" for i in self.path if (os.path.exists(i + ".landscape_tensors.pth"))]
        self.reverse_path = [i + ".reversed_landscape_tensors.pth" for i in self.path if (os.path.exists(i + ".reversed_landscape_tensors.pth"))]
        self.steps_per_epoch = steps_per_epoch
            
    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def __getitem__(self, index):
        reverse = True if random.random() < 0.5 else False
        path = self.reverse_path if reverse else self.original_path
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(path), (1,))[0]
                data_id = (data_id + index) % len(path) # For fixed seed.
                path_tgt = path[data_id]

                # load the condition latent
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))
                cond_idx = random.randint(1, 10)
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)

                # load the target trajectory
                base_path = path_tgt.rsplit('/', 2)[0]
                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                with open(tgt_camera_path, 'r') as file:
                    cam_data = json.load(file)
                
                cam_idx = list(range(81))

                # Intrinsic matrix
                focal = float(base_path.split('train/f')[-1].split('_aperture')[0])
                focal = focal / 23.76  # 23.76: sensor height/width
                intrinsics = torch.tensor([0, focal, focal, 0.5, 0.5, 0, 0]).unsqueeze(0).repeat(len(cam_idx), 1)

                # rel poses
                cond_traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{cond_idx:02d}"]) for idx in cam_idx]
                if reverse:
                    cond_traj = cond_traj[::-1]
                cond_traj = np.stack(cond_traj).transpose(0, 2, 1)
                cond_traj = cond_traj[:, :, [1, 2, 0, 3]]
                cond_traj[:, :3, 1] *= -1.
                cond_traj[:, :3, 3] /= 100
                cond_ref_w2c = np.linalg.inv(cond_traj[0])
                cond_c2ws = torch.as_tensor(cond_ref_w2c[None] @ cond_traj)
                cond_c2ws = rearrange(cond_c2ws[:, :3], 'b c d -> b (c d)')
                cond_cam_params = torch.cat([intrinsics, cond_c2ws], dim=1)

                tgt_traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{tgt_idx:02d}"]) for idx in cam_idx]
                if reverse:
                    tgt_traj = tgt_traj[::-1]
                tgt_traj = np.stack(tgt_traj).transpose(0, 2, 1)
                tgt_traj = tgt_traj[:, :, [1, 2, 0, 3]]
                tgt_traj[:, :3, 1] *= -1.
                tgt_traj[:, :3, 3] /= 100
                # tgt_ref_w2c = np.linalg.inv(tgt_traj[0])
                tgt_c2ws = torch.as_tensor(cond_ref_w2c[None] @ tgt_traj)
                tgt_c2ws = rearrange(tgt_c2ws[:, :3], 'b c d -> b (c d)')
                tgt_cam_params = torch.cat([intrinsics, tgt_c2ws], dim=1)

                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")["y"]
                data = {
                    "input_latents": data_tgt,
                    'traj': tgt_cam_params,
                    'traj_cond': cond_cam_params
                }

                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                data.update(data_cond)
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}, tgt: {path_tgt}, cond: {path_cond}")
                index = random.randrange(len(path))
        return data

    def __len__(self):
        return self.steps_per_epoch
    
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, reverse=False, frame_interval=1, num_frames=81, height=480, width=832):
        metadata = pd.read_csv('dataset/metadata.csv')
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()

        # comment out this part if you want to resume the extraction
        # new_path = []
        # new_text = []
        # for file_path, text in zip(self.path, self.text):
        #     fname = file_path + ".reversed_landscape_tensors.pth" if reverse else file_path + ".landscape_tensors.pth"
        #     if not os.path.exists(fname):
        #         new_path.append(file_path)
        #         new_text.append(text)
        # self.path = new_path
        # self.text = new_text

        self.path = self.path
        self.text = self.text

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.reverse = reverse

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        if self.reverse:
            frames = frames[::-1]

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames


    def __getitem__(self, index):
        while True:
            try:
                path = self.path[index]
                prompt = self.text[index]
                video = self.load_video(path)

                data = {
                    'path': path,
                    'prompt': prompt,
                    'v_cond': video,
                }
                break
            except:
                index += 1

        return data

    def __len__(self):
        return len(self.path)