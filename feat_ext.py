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
        model = model.to("cuda:0")
        feat = model(inputs)

        # adaptive average pooling across the w, h, t axis
        feat = torch.nn.AdaptiveAvgPool3d((1, 1, 1))(feat).squeeze()
        
        # when the last batch has only one sample
        if len(feat.shape) == 1:
            # squeeze operation converts (1, 2304) to (2304)
            # input arrays must have same number of dimensions for np.concatenate
            feat = feat.unsqueeze(0) # (2304) --> (1, 2304)
        feat = feat.cpu().numpy()
        feat_arr += (feat,)

    # concat all feats
    feat_arr = np.concatenate(feat_arr, axis=0)
    return feat_arr


def decode_frames(dataloader_cfg, vid_id):
    # get video file path
    video_file = os.path.join(dataloader_cfg.DATA.PATH_TO_DATA_DIR, vid_id) + dataloader_cfg.DATA.VID_FILE_EXT
    # create output folder
    output_folder = os.path.join(dataloader_cfg.DATA.TMP_FOLDER, vid_id)
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
        "{:s}".format(str(dataloader_cfg.DATA.FPS)),
        "-s",
        "{}x{}".format(dataloader_cfg.DATA.SAMPLE_SIZE[0], dataloader_cfg.DATA.SAMPLE_SIZE[1]),
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


def remove_frames(dataloader_cfg, vid_id):
    output_folder = os.path.join(dataloader_cfg.DATA.TMP_FOLDER, vid_id)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test(dataloader_cfg):
    """
    Perform feature extraction using the pretrained video model.
    Args:
        dataloader_cfg (CfgNode): configs. Details can be found in yaml
    """
    # Set random seed from configs.
    np.random.seed(dataloader_cfg.RNG_SEED)
    torch.manual_seed(dataloader_cfg.RNG_SEED)

    vid_root = os.path.join(dataloader_cfg.DATA.PATH_TO_DATA_DIR, dataloader_cfg.DATA.PATH_PREFIX)
    videos_list_file = os.path.join(dataloader_cfg.DATA.VID_LIST)

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        lines = [line.rstrip() for line in f]
    videos = []
    for line in lines:
        vid, num_frames = line.split(" ")
        videos.append((vid, int(num_frames)))
    print("Done")
    print("----------------------------------------------------------")

    if dataloader_cfg.DATA.READ_VID_FILE:
        rejected_vids = []

    print("{} videos to be processed...".format(len(videos)))
    print("----------------------------------------------------------")

    os.makedirs(dataloader_cfg.OUTPUT_DIR, exist_ok=True)

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
            out_path = os.path.join(dataloader_cfg.OUTPUT_DIR, path)
            # out_file = vid_id.split(".")[0] + ".npz"
            out_file = out_path + ".npz"

            if os.path.exists(out_file):
                print(
                    "{}. {} already exists".format(vid_no, os.path.split(out_file)[-1])
                )
                print("----------------------------------------------------------")
                continue
            if os.path.exists(os.path.join(dataloader_cfg.DATA.TMP_FOLDER, vid_id)):
                print("{}. {} Decoded frames already exist".format(vid_no, vid_id))
                print("==========================================================")
                continue
            print("{}. Decoding {}...".format(vid_no, vid))
            # extract frames from video
            status, msg = decode_frames(dataloader_cfg, vid_id)
            # assert status, msg.decode('utf-8')

            print("{}. Processing {}...".format(vid_no, vid))
            dataset = VideoDataset(dataloader_cfg, dataloader_cfg.DATA.TMP_FOLDER, vid_id, num_frames)
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dataloader_cfg.NUM_GPUS * 2,
                shuffle=False,
                sampler=None,
                num_workers=dataloader_cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=dataloader_cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
            )

            # Perform inference on the current video
            feat_arr = perform_inference(test_loader, model)
            print("{}. Finishing {}...".format(vid_no, vid))

            np.savez(out_file, feats=feat_arr)

            # remove the extracted frames
            remove_frames(dataloader_cfg, vid_id)
            print("Done.")
            print("----------------------------------------------------------")
        except Exception as e:
            print(e, cur_video)
            pass

    if dataloader_cfg.DATA.READ_VID_FILE:
        print("Rejected Videos: {}".format(rejected_vids))

    print("----------------------------------------------------------")


def main():
    """
    Main function to spawn the train and test process.
    """
    parser = argparse.ArgumentParser(description="pytorch video feature extraction")
    parser.add_argument(
        "--dataloader_cfg",
        type=str,
        default="/home/viswa/TAdaConv/configs/epic_kitchens.yaml",
        help="config file to perform feature extraction",
    )

    params = parser.parse_args()
    with open(params.dataloader_cfg, "r") as stream:
        dataloader_cfg = edict(yaml.safe_load(stream))

    print(dataloader_cfg)
    test(dataloader_cfg)


if __name__ == "__main__":
    main()
