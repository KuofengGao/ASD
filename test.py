import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.dataset import PoisonLabelDataset
from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from model.model import LinearModel
from model.utils import (
    get_criterion,
    get_network,
    load_state,
)
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.semi import linear_test

def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/baseline_asd.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",
    )

    args = parser.parse_args()
    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(
        config, inner_dir, config_name, args.resume
    )
    args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
        config, inner_dir, config_name, args.resume
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpu = 0
    set_seed(**config["seed"])
    logger = get_logger(args.log_dir, "asd_test.log", args.resume, gpu == 0)
    torch.cuda.set_device(gpu)

    logger.info("===Prepare data===")
    bd_config = config["backdoor"]
    logger.info("Load backdoor config:\n{}".format(bd_config))
    bd_transform = get_bd_transform(bd_config)
    target_label = bd_config["target_label"]

    pre_transform = get_transform(config["transform"]["pre"])
    train_primary_transform = get_transform(config["transform"]["train"]["primary"])
    train_remaining_transform = get_transform(config["transform"]["train"]["remaining"])
    train_transform = {
        "pre": pre_transform,
        "primary": train_primary_transform,
        "remaining": train_remaining_transform,
    }
    logger.info("Training transformations:\n {}".format(train_transform))
    test_primary_transform = get_transform(config["transform"]["test"]["primary"])
    test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
    test_transform = {
        "pre": pre_transform,
        "primary": test_primary_transform,
        "remaining": test_remaining_transform,
    }
    logger.info("Test transformations:\n {}".format(test_transform))

    logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    )
    
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )

    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])


    logger.info("\n===Setup training===")
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model = linear_model.cuda(gpu)

    criterion = get_criterion(config["criterion"])
    criterion = criterion.cuda(gpu)
    logger.info("Create criterion: {} for test".format(criterion))

    logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    load_state(
        linear_model,
        args.resume,
        args.ckpt_dir,
        gpu,
        logger,
    )
    
    logger.info("Test model on clean data...")
    linear_test(
        linear_model, clean_test_loader, criterion, logger
    )

    logger.info("Test model on poison data...")
    linear_test(
        linear_model, poison_test_loader, criterion, logger
    )

if __name__ == "__main__":
    main()
