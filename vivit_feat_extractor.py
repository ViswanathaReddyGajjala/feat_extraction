"""Extract features for videos using pre-trained networks"""
import numpy as np
import torch
import os
import subprocess
import shutil
import time
import cv2
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
from epic_kitches_dataloader import VideoDataset
import torch.nn as nn
import argparse
from models.base.builder import build_model
from utils.config import Config
import utils.checkpoint as cu

import utils.logging as logging

logger = logging.get_logger(__name__)

torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)


@torch.no_grad()
def perform_inference(test_loader, model):
    """
    Perform mutli-view testing that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
    """
    feat_arr = tuple()
    for inputs in tqdm(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to("cuda:0", non_blocking=True)
        else:
            inputs = inputs.to("cuda:0", non_blocking=True)

        # print([f.shape for f in inputs])
        # preds, feat = model(inputs[0])
        preds, feat_left = model(inputs[0])  # left part
        preds, feat_center = model(inputs[1])
        preds, feat_right = model(inputs[2])
        # print([f.shape for f in [feat_left, feat_center, feat_right]])
        feat = torch.mean(torch.stack([feat_left, feat_center, feat_right]), axis=0)
        # print(feat.shape)

        # when the last batch has only one sample
        if len(feat.shape) == 1:
            feat = feat.unsqueeze(0)  # (768) --> (1, 768)
        feat = feat.cpu().numpy()
        feat_arr += (feat,)

    # concat all feats
    feat_arr = np.concatenate(feat_arr, axis=0)
    print(feat_arr.shape)
    return feat_arr


def decode_frames(cfg, vid_id):
    # get video file path
    video_file = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, vid_id) + cfg.DATA.VID_FILE_EXT
    # create output folder
    output_folder = os.path.join(cfg.DATA.TMP_FOLDER, vid_id)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # ffmpeg cmd
    command = [
        "ffmpeg",
        "-i",
        "{:s}".format(video_file),
        "-vf",
        '"hflip"',
        "-r",
        "{:s}".format(str(cfg.DATA.FPS)),
        "-s",
        "{}x{}".format(cfg.DATA.SAMPLE_SIZE[0], cfg.DATA.SAMPLE_SIZE[1]),
        "-f",
        "image2",
        "-q:v",
        "1",
        "{:s}/%010d.jpg".format(output_folder),
    ]
    command = " ".join(command)

    # call ffmpeg
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return False, err.output

    return True, None


def remove_frames(cfg, vid_id):
    output_folder = os.path.join(cfg.DATA.TMP_FOLDER, vid_id)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test(cfg, model_cfg):
    """
    Perform feature extraction using the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in yaml
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Build the video model and print model statistics.
    model, model_ema = build_model(model_cfg)
    model.cuda()

    model_bucket = None
    cu.load_test_checkpoint(model_cfg, model, model_ema, model_bucket)

    # Enable eval mode.
    model.eval()

    # switch to data parallel
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(cfg.NUM_GPUS)])

    vid_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)
    videos_list_file = os.path.join(cfg.DATA.VID_LIST)

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        lines = [line.rstrip() for line in f]
    videos = []
    for line in lines:
        vid, num_frames = line.split(" ")
        videos.append((vid, int(num_frames)))

    print("Done")
    print("----------------------------------------------------------")

    if cfg.DATA.READ_VID_FILE:
        rejected_vids = []

    print("{} videos to be processed...".format(len(videos)))
    print("----------------------------------------------------------")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    # videos = videos[::-1]
    for vid_no, cur_video in enumerate(videos):
        try:
            vid, num_frames = cur_video
            # Create video testing loaders.
            path_to_vid = os.path.join(vid_root, os.path.split(vid)[0])
            vid_id = os.path.split(vid)[1]
            print("vid_id", vid_id)

            path = os.path.split(vid)[-1]
            out_path = os.path.join(cfg.OUTPUT_DIR, path)
            # out_file = vid_id.split(".")[0] + ".npz"
            out_file = out_path + ".npz"

            if os.path.exists(out_file):
                print(
                    "{}. {} already exists".format(vid_no, os.path.split(out_file)[-1])
                )
                print("----------------------------------------------------------")
                continue
            if os.path.exists(os.path.join(cfg.DATA.TMP_FOLDER, vid_id)):
                print("{}. {} Decoded frames already exist".format(vid_no, vid_id))
                print("==========================================================")
                # continue
            print("{}. Decoding {}...".format(vid_no, vid))
            # extract frames from video
            status, msg = decode_frames(cfg, vid_id)
            # assert status, msg.decode('utf-8')

            print("{}. Processing {}...".format(vid_no, vid))
            dataset = VideoDataset(cfg, cfg.DATA.TMP_FOLDER, vid_id, num_frames)
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.NUM_GPUS * 2,
                shuffle=False,
                sampler=None,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
            )

            # Perform inference on the current video
            feat_arr = perform_inference(test_loader, model)
            print("{}. Finishing {}...".format(vid_no, vid))

            np.savez(out_file, feats=feat_arr)

            # remove the extracted frames
            remove_frames(cfg, vid_id)
            print("Done.")
            print("----------------------------------------------------------")
        except Exception as e:
            print(e, cur_video)
            pass

    if cfg.DATA.READ_VID_FILE:
        print("Rejected Videos: {}".format(rejected_vids))

    print("----------------------------------------------------------")


def main():
    """
    Main function to spawn the train and test process.
    """

    # params = parser.parse_args()
    cfg_path = "configs/epic_kitchens.yaml"
    with open(cfg_path, "r") as stream:
        cfg = edict(yaml.safe_load(stream))

    print(cfg)
    model_cfg = Config(load=True)
    print(model_cfg)
    test(cfg, model_cfg)


if __name__ == "__main__":
    main()
