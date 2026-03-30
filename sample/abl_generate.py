# %%writefile /kaggle/working/mdm/sample/generate.py
# This code is based on https://github.com/openai/guided-diffusion
"""z
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
from numpy import zeros, array, argmin, inf, full
from math import isinf
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil
import shutil
from data_loaders.tensors import collate, t2m_collate
import sys
import math
import numpy as np
import cv2
import os
import _pickle as cPickle
import torch.nn.functional as F
import gzip
import subprocess
import torch
import torch.nn as nn

# from dtw import dtw

# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from numpy.random import random

from scipy.spatial.distance import euclidean
# euclidean_norm = lambda x, y: np.sum(np.abs(x - y))


from scipy.spatial.distance import euclidean
import numpy as np
from numpy import zeros, array, argmin, inf, full
from math import isinf

import pickle
import os
import numpy as np
import torch


def process_sentence_for_slt(text):
    text = text.lower()
    return text.strip()

def process_skel(sign_line, trg_size, des=1):
    full_signs = [(float(joint) - 1e-8) for _, joint in enumerate(sign_line)]
    full_signs = np.array(full_signs)
    full_signs = np.reshape(full_signs, (-1, trg_size))
    if des != 0:
        full_signs = full_signs[:,:-des]
    return full_signs

def transform_tmp_2_slt_data(split_path, data_dim=150, des=0, out_name="phoenix14t"):
    # split_path : ../.../dev
    saved_folder = os.path.dirname(split_path)
    split_set = os.path.basename(split_path)
    saved_file_name = f"{saved_folder}/{out_name}.pami0.{split_set}"

    text_path = f"{split_path}.text"
    sign_path = f"{split_path}.skels"
    data_list = []
    with open(text_path, "r") as text_file:
        with open(sign_path, "r") as sign_file:
            for i, (txt_line, sign_line) in enumerate(zip(text_file, sign_file)):
                sign_line = sign_line.split()
                if len(sign_line) % data_dim != 0:
                    print(f"len(sign_line) : {len(sign_line)}")
                    continue
                sign_inp = process_skel(sign_line, trg_size=data_dim, des=des)
                txt_inp = process_sentence_for_slt(txt_line)
                item = {
                    "name": f"name_{i}",
                    "signer": f"signer_{i}",
                    "gloss": txt_inp,
                    "text": txt_inp,
                    "sign": sign_inp,
                }
                data_list.append(item)

    with open(saved_file_name, 'wb') as file:
        pickle.dump(data_list, file)
    print(f"Done back translation file for {split_set} !")


import shutil
from data_loaders.get_data import get_dataset_loader


from tqdm import tqdm
def main():
    args = generate_args()

    assert args.guidance_param == 3.5, "Ablation requires guidance_param=3.5"
   
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 500
    fps = 25

    # Không học dự đoán được số lượng frames cho câu
    # n_frames = min(max_frames, int(args.motion_length*fps))
    n_frames = max_frames

    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.dirname(args.model_path) + "/back_translation"

    # Check if the folder exists and remove it
    if os.path.exists(out_path):
        shutil.rmtree(out_path)  # Removes the directory and all its contents
    os.makedirs(out_path, exist_ok=True)

    # -------------------- LOAD DATA ----------------------------
    print(f'Loading dataset for back translation [BATCH_SIZE = {args.batch_size}]...')


    # val_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=n_frames, split='val')
    test_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=n_frames, split='test')

    data_dict = {"test":test_data}



    # -------------------- LOAD MODEL ---------------------------
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(args, test_data)


    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)


    if args.guidance_param != 1:
        print(f"********************************* USE CLASSIFIER with guidance_param={args.guidance_param}")
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    else:
        print(f"********************************* NOOOO guidance_param={args.guidance_param}")
    
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_dtws = []
    for data_name, data in data_dict.items():
        with open(f"{out_path}/{data_name}.skels", "w") as skel_file:
            with open(f"{out_path}/{data_name}.text", "w") as text_file:
                for k, (ground_truth, model_kwargs) in enumerate(data):
                    if args.guidance_param != 1:
                        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
                    sample_fn = diffusion.p_sample_loop
                    sample = sample_fn(
                        model,
                        (args.batch_size, model.in_channels, 1, ground_truth.shape[-1]),  # BUG FIX
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=True,
                        noise=None,
                    )
                
                    # Z - NORM
                    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                    sample = sample.squeeze(1)
                
                    texts = model_kwargs['y']['text']
                    
                    lengths = model_kwargs['y']['lengths'].to("cuda")
                    texts = model_kwargs['y']['text']
                    # model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])


                    ground_truth = data.dataset.t2m_dataset.inv_transform(ground_truth.cpu().permute(0, 2, 3, 1)).float()
                    ground_truth = ground_truth.squeeze(1)

                    batch_dtw = []
                    
                    for i, (caption, gt_motion, motion, gt_len) in enumerate(zip(texts, ground_truth, sample, lengths)):
                        motion = motion[:gt_len].cpu().numpy()
                        gt_motion = gt_motion[:gt_len].cpu().numpy()

                        _, _, dis_dtw = alter_DTW_timing(motion / 3, gt_motion / 3, transform_pred=False)
                        batch_dtw.append(dis_dtw)
    
                        text_file.write(f"{caption.strip()}\n")
                        
                        hyp_seq = ' '.join(map(str, motion.flatten())).strip()
                        skel_file.write(f"{hyp_seq}\n")

                    m_dtw = np.array(batch_dtw).mean()
                    all_dtws.append(m_dtw)
                        
                transform_tmp_2_slt_data(f"{out_path}/{data_name}", data_dim=150, des=0, out_name=f"{data_name}_back_translation")
                with open(f"{out_path}/dtw.txt", "w") as f:
                    for i, score in enumerate(all_dtws):
                        f.write(f"Batch {i}| {score}\n")
                    dtw = np.array(all_dtws).mean()
                    f.write(f"Total | {dtw}\n")

    os.remove(f"{out_path}/{data_name}.skels")
    os.remove(f"{out_path}/{data_name}.text")
    print(f"Evaluation results at {out_path}.")
    print(f"******************** DTW={round(dtw, 3)}")

 

# Apply DTW
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame

# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq,transform_pred=True):

    # act1 = act1.cpu().numpy()
    # act2 = act2.cpu().numpy()


    
    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))


    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq, pred_seq, dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]
    
    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    
    
    if transform_pred:
        # j tracks the position in the reference sequence
        j = 0
        skips = 0
        squeeze_frames = []
        for (i, pred_num) in enumerate(path[0]):

            if i == len(path[0]) - 1:
                break

            if path[1][i] == path[1][i + 1]:
                skips += 1

            # If a double coming up
            if path[0][i] == path[0][i + 1]:
                squeeze_frames.append(pred_seq[i - skips])
                j += 1
            # Just finished a double
            elif path[0][i] == path[0][i - 1]:
                new_pred_seq[pred_num] = avg_frames(squeeze_frames)
                squeeze_frames = []
            else:
                new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None,
              lengths=None):
    
    # # KKT
    # print("Line 23 in plot_videos.py => Save validation video")
    # ##############

    # Create video template
    FPS = (25 // skip_frames)
    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    elif references is not None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)  # Long

    num_frames = 0
    # print(f"JOINT SHAPE : {joints.shape}")

    for (j, frame_joints) in enumerate(joints):

        frame = np.ones((650, 650, 3), np.uint8) * 255
        
#         frame_joints = frame_joints[:-1] # * 3

        # Reduce the frame joints down to 2D for visualisation - Frame joints 2d shape is (48,2)
        frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]

        # Draw the frame given 2D joints
        draw_frame_2D(frame, frame_joints_2d)

        pred_text = f"PRED LENGTH = {lengths[0]}"
        cv2.putText(frame, pred_text, (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 2)
        # cv2.putText(frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             (0, 0, 0), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            # Cut off the percent_tok and multiply each joint by 3 (as was reduced in training files)
#             ref_joints = ref_joints[:-1] # * 3

            # Reduce the frame joints down to 2D- Frame joints 2d shape is (48,2)
            ref_joints_2d = np.reshape(ref_joints, (50, 3))[:, :2]

            # Draw these joints on the frame
            draw_frame_2D(ref_frame, ref_joints_2d)
            ref_text = f"GT LENGTH = {lengths[1]}"
            cv2.putText(ref_frame, ref_text, (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)
        
        sequence_ID_write = "" + sequence_ID.split("/")[-1]
        cv2.putText(frame, sequence_ID_write, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


        # Write the video frame
        video.write(frame)
        num_frames += 1
        
    # Release the video
    video.release()

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure():

    return (
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        # (3, 4, 3),
        # Changed to avoid wrist, go straight to hands
        (3, 29, 3),

        # right shoulder
        (1, 5, 1),

        # right arm
        (5, 6, 2),
        # (6, 7, 3),
        # Changed to avoid wrist, go straight to hands
        (6, 8, 3),

        # left hand - wrist
        # (7, 8, 4),

        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),

        # left hand - 1st finger
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),

        # left hand - 2nd finger
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),

        # left hand - 3rd finger
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),

        # left hand - 4th finger
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),

        # left hand - 5th finger
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),

        # right hand - wrist
        # (4, 29, 4),

        # right hand - palm
        (29, 30, 5),
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),

        # right hand - 1st finger
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),

        # right hand - 2nd finger
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),

        # right hand - 3rd finger
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),

        # right hand - 4th finger
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),

        # right hand - 5th finger
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24),
    )

# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:

        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))

        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)

        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))

        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints):
    # Line to be between the stacked
#     draw_line(frame, [1, 650], [1, 1], c=(0,0,0), t=1, width=1)




    # Give an offset to center the skeleton around
    offset = [350, 250]

    # Get the skeleton structure details of each bone, and size
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]

    # Increase the size and position of the joints
    joints = joints * 10 * 12 * 2
    joints = joints + np.ones((50, 2)) * offset

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):

        c = get_bone_colour(skeleton,j)

        draw_line(frame, [joints[skeleton[j, 0]][0], joints[skeleton[j, 0]][1]],
                  [joints[skeleton[j, 1]][0], joints[skeleton[j, 1]][1]], c=c, t=1, width=1)

# get bone colour given index
def get_bone_colour(skeleton,j):
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2 and skeleton[j, 1] == 3:  # left arm
        c = (0, 102, 204)
    elif bone == 3 and skeleton[j, 0] == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 2 and skeleton[j, 1] == 6:  # right arm
        c = (0, 153, 0)
    elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
        c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)
    return c


if __name__ == "__main__":
    main()



# %cd /kaggle/working/mdm
# !python -m sample.generate \
# --model_path /kaggle/working/mdm/save/qna-smooth-l1loss-vel-lvMASK/model000250000.pt \
# --output_dir /kaggle/working/mdm/save/qna-smooth-l1loss-vel-lvMASK/250k \
# --guidance_param 3.5 \
# --batch_size 64