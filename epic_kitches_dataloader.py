#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified to load and process frames of a single video

import os
import glob
import torch
import torch.utils.data
import numpy as np
import cv2


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH == "slowfast":
        fast_pathway = frames
        """
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        """
        frame_list = [fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(cfg.MODEL.ARCH, ["slowfast"])
        )
    return frame_list


class VideoDataset(torch.utils.data.Dataset):
    """
    Construct the untrimmed video loader, then sample
    segments from the videos. The videos are segmented by centering
    each frame as per the output size i.e. cfg.DATA.NUM_FRAMES.
    """

    def __init__(self, cfg, vid_path, vid_id, num_frames):
        """
        Construct the video loader for a given video.
        Args:
            cfg (CfgNode): configs.
            vid_path (string): path to the video
            vid_id (string): video name
        """
        self.cfg = cfg

        self.vid_path = vid_path
        self.vid_id = vid_id

        self.stride = cfg.DATA.STRIDE
        self.out_size = cfg.DATA.NUM_FRAMES
        self.step_size = cfg.DATA.SAMPLING_RATE

        # frame folder
        self.frame_folder = os.path.join(vid_path, vid_id)
        print(self.frame_folder)
        assert os.path.exists(self.frame_folder)

        if isinstance(cfg.DATA.SAMPLE_SIZE, list):
            self.sample_width, self.sample_height = cfg.DATA.SAMPLE_SIZE
        elif isinstance(cfg.DATA.SAMPLE_SIZE, int):
            self.sample_width = self.sample_height = cfg.DATA.SAMPLE_SIZE
        else:
            raise Exception(
                "Error: Frame sampling size type must be a list [Height, Width] or int"
            )

        # list all frames
        self.frame_list = sorted(glob.glob(os.path.join(self.frame_folder, "*.jpg")))
        self.num_frames = len(self.frame_list)

        # check frame folder
        
        if self.num_frames != num_frames:
            print(
                "{:s} has {:d} frames expecting {:d} frames".format(
                    vid_id, self.num_frames, num_frames
                )
            )

    def _process_frame(self, arr):
        """
        Pre process an array
        Args:
            arr (ndarray): an array of frames or a ndarray of an image
                of shape T x H x W x C or W x H x C respectively
        Returns:
            arr (tensor): a normalized torch tensor of shape C x T x H x W
                or C x W x H respectively
        """
        arr = torch.from_numpy(arr).float()

        # Normalize the values
        arr = arr / 255.0
        arr = arr - torch.tensor(self.cfg.DATA.MEAN)
        arr = arr / torch.tensor(self.cfg.DATA.STD)

        # T H W C -> C T H W.
        if len(arr.shape) == 4:
            arr = arr.permute(3, 0, 1, 2)
        elif len(arr.shape) == 3:
            arr = arr.permute(2, 0, 1)

        return arr

    def _pad_to_length(self, clip):
        num_frames = self.out_size
        assert clip.shape[0] <= num_frames
        # no padding needed
        if clip.shape[0] == num_frames:
            return clip
        # prep for padding
        new_clip = np.zeros(
            [num_frames, clip.shape[1], clip.shape[2], clip.shape[3]], dtype=clip.dtype
        )
        clip_num_frames = clip.shape[0]
        padded_num_frames = num_frames - clip_num_frames

        # fill in new_clip (repeating the last frame)
        new_clip[0:clip_num_frames, :, :, :] = clip[:, :, :, :]
        new_clip[clip_num_frames:, :, :, :] = np.tile(
            clip[-1, :, :, :], (padded_num_frames, 1, 1, 1)
        )
        return new_clip

    def __getitem__(self, index):

        start = int(index * self.stride)
        end = int(index * self.stride + self.step_size * self.out_size)
        end = min(end, self.num_frames - 1)

        # we assume that the video is already downsampled
        img_list = tuple()
        for idx in range(start, end, self.step_size):
            img = cv2.imread(self.frame_list[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR --> RGB
            img_list += (img[np.newaxis, :, :, :],)

        # T H W C
        decoded_frames = np.concatenate(img_list, axis=0)
        # T H W C (this will always return an array using newly allocated mem)
        clip = self._pad_to_length(decoded_frames)
        # BGR & -> C T H W
        clip = self._process_frame(np.ascontiguousarray(clip))
        # create the pathways
        frame_list = pack_pathway_output(self.cfg, clip)
        return frame_list

    def __len__(self):
        """
        Returns:
            (int): the number of frames in the video.
        """
        return self.num_frames // self.stride


# out_size = 10
# num_frames = 36000
# stride = 10
# step_size = 1


# len = num_frames // stride

# start = int(index * stride)
# end = int(index * stride + step_size * out_size)
# end = min(end, num_frames - 1)

# # we assume that the video is already downsampled
# j =0
# for idx in range(start, end, step_size):
#     j+=1
# j
