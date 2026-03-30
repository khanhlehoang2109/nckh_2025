# %%writefile /kaggle/working/mdm/exp_code/sample/generate.py


# This code is based on https://github.com/openai/guided-diffusion
"""
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
from model.cfg_sampler import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
import shutil
from data_loaders.tensors import collate
import sys
import math
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torch
# from dtw import dtw

# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm


import numpy as np
from numpy import zeros, array, argmin, inf, full
from math import isinf


from loguru import logger
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
import smplx
from diffusers import DDPMPipeline




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

def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame
# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq,transform_pred=True):

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


def dtw_kkt(pred_seq,ref_seq,ref_len):
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:ref_len,:], pred_seq[:ref_len,:], dist=euclidean_norm)
    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]
    return d

def process_sentence(text):
    text = text.lower()
    return text.strip()

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid




ign_idx = [0,1,2,4,5,7,8,10,11]
above_body_idx = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
body_pose_num_keypoints=21

def get_full_body_pose(above_body):
    # above_body.shape = t, 12 x 3
    T = above_body.shape[0]
    above_body = above_body.reshape(T, len(above_body_idx), 3)
    full_body = torch.zeros(T, body_pose_num_keypoints, 3)
    full_body[:, above_body_idx] = above_body
    full_body = full_body.reshape(T, -1)
    return full_body

def split_keypoint(pose):
    above_body = pose[..., :12*3]
    body_pose = get_full_body_pose(torch.tensor(above_body))
    cur = 12*3

    lhand_pose = pose[..., cur: cur + 15*3]
    cur = cur + 15*3
    
    rhand_pose = pose[..., cur: cur + 15*3]
    cur = cur + 15*3

    expression = pose[..., cur: cur + 10]
    cur = cur + 10

    jaw_pose = pose[..., cur: ]

    return None, None, body_pose, torch.tensor(lhand_pose), \
    torch.tensor(rhand_pose), \
    None, None, torch.tensor(expression), \
    torch.tensor(jaw_pose), None


def main():
    split = "test"

    args = generate_args()

    assert args.guidance_param == 5.5, "Inference on How2Sign requires guidance_param=5.5"



    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 500
    fps = 25
    n_frames = max_frames = min(max_frames, int(args.motion_length*fps))
    print(f"======================= NUM OF FRAME = {n_frames}")
    
    
    # ================================ LOAD BODY MODEL ================================================================
    model_folder = "/kaggle/input/smlpx-models/models"
    model_type='smplx'
    ext='npz'
    gender='male'
    num_betas=10
    num_expression_coeffs=10
    use_face_contour=False
    
    body_model = smplx.create(model_folder, model_type=model_type,
                        gender=gender, use_face_contour=use_face_contour,
                        num_betas=num_betas,
                        num_expression_coeffs=num_expression_coeffs,
                        ext=ext,
                        use_pca=False)# .to("cuda")
    # ================================================================================================
    
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')[:50]
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')[:50]

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        raw_texts = [args.text_prompt]
        texts = []
        for text in raw_texts:
            texts.append(process_sentence(text))
        args.num_samples = 1
        
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples





    print('Loading dataset...')

    if is_using_data:
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=n_frames, split=split)
    else:
        data = load_dataset(args, n_frames, n_frames, split=split)
        
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    # args.diffusion_steps = 250




    
    model, diffusion = create_model_and_diffusion(args, data)


    
    # Load trained DDPM model
    
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
    model.requires_grad_(False)
    all_gt_motions = []

    if is_using_data:
        iterator = iter(data)
        ground_truth, model_kwargs = next(iterator)

        n_frames = ground_truth.shape[-1]
        

        ground_truth = data.dataset.t2m_dataset.inv_transform(ground_truth.cpu().permute(0, 2, 3, 1)).float()
        ground_truth = ground_truth.squeeze(1)

        
        all_gt_motions.append(ground_truth)
        gt_lengths = model_kwargs["y"]["lengths"].cpu().numpy()
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
        sample_fn = diffusion.ddim_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.in_channels, 1, n_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            noise=None,
        )

        

        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()

#         bs, feat, frames, dim = sample.shape
        sample = sample.squeeze(1)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            try:
                text_key = 'file'
                all_text += model_kwargs['y'][text_key]
            except:
                text_key = 'text'
                all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())

        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]

    try:
      all_gt_motions = np.concatenate(all_gt_motions, axis=0)
      all_gt_motions = all_gt_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    except:
      all_gt_motions = None


    all_text = all_text[:total_num_samples]


    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        
    os.makedirs(out_path)


    print(f"saving visualizations to [{out_path}]...")

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)


    for sample_i in range(args.num_samples):
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i]
            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)

            if all_gt_motions is not None:
                gt_motion = all_gt_motions[sample_i]
                gt_len = gt_lengths[sample_i]
                
                _, _, dis_dtw = alter_DTW_timing(motion / 3, gt_motion / 3, transform_pred=False)
                # batch_dtw.append(dis_dtw)
                caption = f"DTW = {round(dis_dtw, 2)}"
            
                motion = motion[:gt_len]
                gt_motion = gt_motion[:gt_len]
                plot_video(body_model, 
                           joints=motion,
                      file_path=out_path,
                      video_name=save_file,
                      references=gt_motion,
                      skip_frames=1,
                      sequence_ID=caption,
                          fps=fps)
                    

            else:

                plot_video(body_model, joints=motion,
                          file_path=out_path,
                          video_name=save_file,
                          references=None,
                          skip_frames=1,
                          sequence_ID=caption,
                          fps=fps)
                

            # print(f"[{caption} | {animation_save_path}]")

    abs_path = os.path.abspath(out_path)

    print(f'[Done] Results are at [{abs_path}]')

    
def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files

# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:

        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))

        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)

        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))

        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames, split):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split=split,
                              hml_mode='text_only')
    return data


def get_body_params(joints, body_model):
    
    _, _, body_pose, left_hand_pose, right_hand_pose, \
    _, _, expression, jaw_pose, _ = split_keypoint(joints)

    leye_pose = torch.zeros(body_pose.shape[0], 3)
    reye_pose = torch.zeros(body_pose.shape[0], 3)

    root = torch.zeros(body_pose.shape[0], 3)
    root[..., :1] += 0.25
    
    betas = torch.randn([1, body_model.num_betas], dtype=torch.float32)

    body_params = {"betas":betas, "global_orient":root, \
            "body_pose":body_pose, "left_hand_pose":left_hand_pose, \
               "right_hand_pose":right_hand_pose, "jaw_pose":jaw_pose, \
               "leye_pose":leye_pose, "reye_pose":reye_pose, \
               "expression":expression}
    return body_params
    
# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(body_model,
                joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None,
              fps=25):
        
        
        # Create video template
        FPS = (fps // skip_frames)
        video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
        

        if references is None:
            
            img_w, img_h = 800, 800
            
            body_params = get_body_params(joints, body_model)
#             print(f"===> =body_params : \n{body_params}")
            img_array = render_smpl_params(body_model, body_parms=body_params, imw=img_w, imh=img_h, output_obj_folder=f"{file_path}/objs")[None, None]
            imagearray2file(img_array, outpath=video_file, fps=FPS)
            
            logger.success(f'Inference [{sequence_ID}] finish at | {video_file}')
            

        else:
            
            img_w, img_h = 1300, 650
            
            pred_body_params = get_body_params(joints, body_model)
            pred_img_array = render_smpl_params(body_model, body_parms=pred_body_params, imw=img_h, imh=img_h)
            
            ref_body_params = get_body_params(references, body_model)
            ref_img_array = render_smpl_params(body_model, body_parms=ref_body_params, imw=img_h, imh=img_h)
            
            
# Ground truth : (390, 139)
# Prediction : (500, 139)
# Ground truth in plot function: (1, 1, 390, 1300, 1300, 3)
# Prediction in plot function: (1, 1, 390, 1300, 1300, 3)
            
            
            # print(f"Ground truth in plot function: {ref_img_array.shape}")
            # print(f"Prediction in plot function: {pred_img_array.shape}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_file, fourcc, float(FPS), (img_w, img_h), True)  # Long
            
            for pred_frame, ref_frame in zip(pred_img_array, ref_img_array):
                
                # Split line
                draw_line(pred_frame, [1, img_h], [1, 1], c=(0,0,0), t=1, width=1)

                # Anotate output
                cv2.putText(pred_frame, "Prediction", (190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
                
                cv2.putText(ref_frame, "Ground Truth", (190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2)
                
                # Concatnate 2 frames
                frame = np.concatenate((pred_frame, ref_frame), axis=1)
                if sequence_ID is not None:
                    cv2.putText(frame, sequence_ID, (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 2)
                
                # Write frame to video
                video.write(frame)
                
            
            # Release the video
            video.release()
            logger.success(f'Evaluate finish at | {video_file}')



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

if __name__ == "__main__":
    main()

# %cd /kaggle/working/mdm/exp_code
# !python -m sample.generate --model_path /kaggle/working/mdm/exp_code/save/how2sign_OURs_no_drop/model000400000.pt \
# --num_repetitions 1 \
# --guidance_param 5.5 \
# --motion_length 4 \
# --text_prompt "You need to stay with something fairly professional."