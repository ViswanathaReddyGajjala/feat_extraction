#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import json

import utils.bucket as bu
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import TestMeter, EpicKitchenMeter
from tqdm import tqdm

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
        # model = model.to("cuda:0")
        # feat = model(inputs)
        preds, feat = model(inputs[0])
        # when the last batch has only one sample
        if len(feat.shape) == 1:
            # squeeze operation converts (1, 2304) to (2304)
            # input arrays must have same number of dimensions for np.concatenate
            feat = feat.unsqueeze(0) # (2304) --> (1, 2304)
        feat = feat.cpu().numpy()
        feat_arr += (feat,)

    # concat all feats
    feat_arr = np.concatenate(feat_arr, axis=0)
    print(feat_arr.shape)
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
        print(err)
        return False, err.output

    return True, None


def remove_frames(dataloader_cfg, vid_id):
    output_folder = os.path.join(dataloader_cfg.DATA.TMP_FOLDER, vid_id)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    Perform multi-view test on the specified test set, where {cfg.TEST.NUM_ENSEMBLE_VIEWS}
    clips and {cfg.TEST.NUM_SPATIAL_CROPS} crops are sampled temporally and spatially, forming 
    in total cfg.TEST.NUM_ENSEMBLE_VIEWS x cfg.TEST.NUM_SPATIAL_CROPS views.
    The softmax scores are aggregated by summation. 
    The predictions are compared with the ground-truth labels and the accuracy is logged.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (Config): The global config object.
    """
    # Enable eval mode.
    os.makedirs(cfg.FEATURES.OUTPUT_PATH, exist_ok=True)

    model.eval()
    test_meter.iter_tic()
    res_dic = {}
    for cur_iter, (inputs, labels, video_idx, meta) in tqdm(enumerate(test_loader), total=len(test_loader)):
        if cur_iter >= 1:
            break
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            for k, v in inputs.items():
                if not isinstance(v, (torch.Tensor, list)):
                    continue
                if isinstance(inputs[k], list):
                    for i in range(len(inputs[k])):
                        inputs[k][i] = v[i].cuda(non_blocking=True)
                else:
                    inputs[k] = v.cuda(non_blocking=True)

            # Transfer the labels to the current GPU device.
            if isinstance(labels["supervised"], dict):
                for k, v in labels["supervised"].items():
                    labels["supervised"][k] = v.cuda()
            else:
                labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()
            if cfg.PRETRAIN.ENABLE:
                for k, v in labels["self-supervised"].items():
                    labels["self-supervised"][k] = v.cuda(non_blocking=True)
            """
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            """
        if cfg.PRETRAIN.ENABLE:
            # currently supports MoSI test.
            preds, _ = model(inputs)
            if misc.get_num_gpus(cfg) > 1:
                pred, label, video_idx = du.all_gather(
                    [preds["move_joint"], labels["self-supervised"]['move_joint'].reshape(preds["move_joint"].shape[0]), video_idx]
                )
            if misc.get_num_gpus(cfg):
                pred = pred.cpu()
                label = label.cpu()
                video_idx = video_idx.cpu()
            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                pred.detach(), label.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)
        else:
            # Perform the forward pass.
            # preds, features = model(inputs) # torch.Size([1, 3, 32, 320, 320])

            # Code from slowfast repo
            # Set random seed from configs.
            with open("/home/viswa/TAdaConv/configs/epic_kitchens.yaml", "r") as stream:
                dataloader_cfg = edict(yaml.safe_load(stream))

            np.random.seed(dataloader_cfg.RNG_SEED)
            torch.manual_seed(dataloader_cfg.RNG_SEED)

            vid_root = os.path.join(dataloader_cfg.DATA.PATH_TO_DATA_DIR, dataloader_cfg.DATA.PATH_PREFIX)
            videos_list_file = os.path.join(dataloader_cfg.DATA.VID_LIST)

            print("Loading Video List ...")
            with open(videos_list_file) as f:
                lines = [line.rstrip() for line in f]
            videos = []
            for line in lines:
                vid = line.split(" ")[0]
                vid = vid.split("/")[-1].split(".")[0]
                num_frames = 25000 # random number to avoid code changes
                videos.append((vid, int(num_frames)))
            print("Done")
            print("----------------------------------------------------------")

            if dataloader_cfg.DATA.READ_VID_FILE:
                rejected_vids = []

            print("{} videos to be processed...".format(len(videos)))
            print("----------------------------------------------------------")

            os.makedirs(dataloader_cfg.OUTPUT_DIR, exist_ok=True)
            os.makedirs(dataloader_cfg.DATA.TMP_FOLDER, exist_ok=True)

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
                        # continue
                    print("{}. Decoding {}...".format(vid_no, vid))
                    # extract frames from video
                    status, msg = decode_frames(dataloader_cfg, vid_id) # @TODO
                    # assert status, msg.decode('utf-8')

                    print("{}. Processing {}...".format(vid_no, vid))
                    dataset = VideoDataset(dataloader_cfg, dataloader_cfg.DATA.TMP_FOLDER, vid_id, num_frames)
                    test_loader_2 = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=dataloader_cfg.NUM_GPUS * 2,
                        shuffle=False,
                        sampler=None,
                        num_workers=dataloader_cfg.DATA_LOADER.NUM_WORKERS,
                        pin_memory=dataloader_cfg.DATA_LOADER.PIN_MEMORY,
                        drop_last=False,
                    )

                    # Perform inference on the current video
                    feat_arr = perform_inference(test_loader_2, model)
                    # preds, feat_arr = model(inputs)
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



            """
            # print(features.shape)
            for path_index, video_path in enumerate(meta['path']):
                feature_path = os.path.join(cfg.FEATURES.OUTPUT_PATH, video_path.split('/')[-1] + ".pt")
                # print(features[path_index].shape)
                torch.save(features[path_index], feature_path)
            # feature_path = os.path.join(cfg.FEATURES.OUTPUT_PATH, meta['path'].split('/')[-1] + ".pt")
            # torch.save(features, feature_path)

            if cfg.DATA.MULTI_LABEL:
                # Mainly for the EPIC-KITCHENS dataset.
                if misc.get_num_gpus(cfg) > 1:
                    preds_verb, preds_noun, labels_verb, labels_noun, video_idx = du.all_gather(
                        [
                            preds["verb_class"], 
                            preds["noun_class"], 
                            labels["supervised"]["verb_class"], 
                            labels["supervised"]["noun_class"], 
                            video_idx
                        ]
                    )
                else:
                    preds_verb  = preds["verb_class"]
                    preds_noun  = preds["noun_class"]
                    labels_verb = labels["supervised"]["verb_class"]
                    labels_noun = labels["supervised"]["noun_class"]
                if misc.get_num_gpus(cfg):
                    preds_verb  = preds_verb.cpu()
                    preds_noun  = preds_noun.cpu()
                    labels_verb = labels_verb.cpu()
                    labels_noun = labels_noun.cpu()
                    video_idx   = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds_verb.detach(),
                    preds_noun.detach(),
                    labels_verb.detach(),
                    labels_noun.detach(),
                    video_idx.detach(),
                    [test_loader.dataset._get_sample_info(i)["name"] for i in video_idx.tolist()] if "name" in test_loader.dataset._get_sample_info(0).keys() else []
                )
                test_meter.log_iter_stats(cur_iter)
            else:

                # Gather all the predictions across all the devices to perform ensemble.
                if misc.get_num_gpus(cfg) > 1:
                    preds, labels_supervised, video_idx = du.all_gather(
                        [preds, labels["supervised"], video_idx]
                    )
                else:
                    labels_supervised = labels["supervised"]
                if misc.get_num_gpus(cfg):
                    preds = preds.cpu()
                    labels_supervised = labels_supervised.cpu()
                    video_idx = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach(), labels_supervised.detach(), video_idx.detach()
                )
                test_meter.log_iter_stats(cur_iter)
            """
        test_meter.iter_tic()

    # save epic-kitchens statistics
    if "epickitchen100" in cfg.TEST.DATASET:
        if cfg.DATA.MULTI_LABEL or not hasattr(cfg.DATA, "TRAIN_VERSION"):
            verb = test_meter.video_preds["verb_class"]
            noun = test_meter.video_preds["noun_class"]
            file_name_verb = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            file_name_noun = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(verb, file_name_verb)
            torch.save(noun, file_name_noun)

            logger.info(
                "Successfully saved verb and noun results to {} and {}.".format(file_name_verb, file_name_noun)
            )
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_verb":
            file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(test_meter.video_preds, file_name)
            logger.info(
                "Successfully saved verb results to {}.".format(file_name)
            )
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_noun":
            file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(test_meter.video_preds, file_name)
            logger.info(
                "Successfully saved noun results to {}.".format(file_name)
            )

    test_meter.finalize_metrics()
    test_meter.reset()


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (Config): The global config object.
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Test with config:")
        logger.info(cfg)

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)

    # Create video testing loaders.
    test_loader = build_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    cfg.LOG_PERIOD = max(len(test_loader) // 10, 5)
    if cfg.DATA.MULTI_LABEL or hasattr(cfg.DATA, "TRAIN_VERSION"):
        test_meter = EpicKitchenMeter(
            cfg,
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
        )
    else:
        test_meter = TestMeter(
            cfg,
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Perform multi-view test on the entire dataset.
    test_meter.set_model_ema_enabled(False)
    perform_test(test_loader, model, test_meter, cfg)
    if model_ema is not None:
        test_meter.set_model_ema_enabled(True)
        perform_test(test_loader, model_ema.module, test_meter, cfg)
    
    # upload results to bucket
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        result_file_name = cfg.TEST.LOG_FILE
        result_file_name = result_file_name.split('.')[0] + "_res" + ".json"
        filename = os.path.join(cfg.OUTPUT_DIR, result_file_name)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        result_file_name = cfg.TEST.LOG_FILE
        result_file_name = result_file_name.split('.')[0] + "_res_ema" + ".json"
        filename = os.path.join(cfg.OUTPUT_DIR, result_file_name)
        if os.path.exists(filename):
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb_ema.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb_ema.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun_ema.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun_ema.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )

    # synchronize all processes on different GPUs to prevent collapsing
    du.synchronize()