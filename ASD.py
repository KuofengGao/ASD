import argparse
import os
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from data.dataset import PoisonLabelDataset, MixMatchDataset
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
    get_optimizer,
    get_scheduler,
    load_state,
)
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.log import result2csv
from utils.trainer.semi import mixmatch_train, linear_test, poison_linear_record

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
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-port",
        default="23456",
        type=str,
        help="port used to set up distributed training",
    )
    args = parser.parse_args()

    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.saved_dir)
    args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.storage_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        args.distributed = True
    else:
        args.distributed = False
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("Distributed training on GPUs: {}.".format(args.gpu))
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, config),
        )
    else:
        print("Training on a single GPU: {}.".format(args.gpu))
        main_worker(0, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    set_seed(**config["seed"])
    logger = get_logger(args.log_dir, "asd.log", args.resume, gpu == 0)
    torch.cuda.set_device(gpu)
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:{}".format(args.dist_port),
            world_size=args.world_size,
            rank=args.rank,
        )
        logger.warning("Only log rank 0 in distributed training!")

    logger.info("===Prepare data===")
    bd_config = config["backdoor"]
    logger.info("Load backdoor config:\n{}".format(bd_config))
    bd_transform = get_bd_transform(bd_config)
    target_label = bd_config["target_label"]
    poison_ratio = bd_config["poison_ratio"]

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
    clean_train_data = get_dataset(
        config["dataset_dir"], train_transform, prefetch=config["prefetch"]
    )
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    )
    
    poison_idx_path = os.path.join(args.saved_dir, "poison_idx.npy")
    if os.path.exists(poison_idx_path):
        poison_train_idx = np.load(poison_idx_path)
        logger.info("Load poisoned index to {}".format(poison_idx_path))
    else:
        poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
        np.save(poison_idx_path, poison_train_idx)
        logger.info("Save poisoned index to {}".format(poison_idx_path))
    

    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label
    )
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )

    poison_train_loader = get_loader(poison_train_data, config["loader"], shuffle=True)
    poison_eval_loader = get_loader(poison_train_data, config["loader"])
    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])


    logger.info("\n===Setup training===")
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model = linear_model.cuda(gpu)
    if args.distributed:
        linear_model = DistributedDataParallel(linear_model, device_ids=[gpu])


    criterion = get_criterion(config["criterion"])
    criterion = criterion.cuda(gpu)
    logger.info("Create criterion: {} for test".format(criterion))

    split_criterion = get_criterion(config["split"]["criterion"])
    split_criterion = split_criterion.cuda(gpu)
    logger.info("Create criterion: {} for data split".format(split_criterion))

    semi_criterion = get_criterion(config["semi"]["criterion"])
    semi_criterion = semi_criterion.cuda(gpu)
    logger.info("Create criterion: {} for semi-training".format(semi_criterion))


    optimizer = get_optimizer(linear_model, config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))
    
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    resumed_epoch, best_acc, best_epoch = load_state(
        linear_model,
        args.resume,
        args.ckpt_dir,
        gpu,
        logger,
        optimizer,
        scheduler,
        is_best=True,
    )

    # clean seed samples
    clean_data_info = {}
    all_data_info = {}
    for i in range(config['num_classes']):
        clean_data_info[str(i)] = []
        all_data_info[str(i)] = []
    for idx, item in enumerate(poison_train_data):
        if item['poison'] == 0:
            clean_data_info[str(item['target'])].append(idx)
        all_data_info[str(item['target'])].append(idx)
    indice = []
    for k, v in clean_data_info.items():
        choice_list = np.random.choice(v, replace=False, size=config["global"]["seed_num"]).tolist()
        indice = indice + choice_list
        all_data_info[k] = [x for x in all_data_info[k] if x not in choice_list]
    indice = np.array(indice)


    choice_num = 0
    for epoch in range(resumed_epoch, config["num_epochs"]):
        logger.info(
            "===Epoch: {}/{}===".format(epoch + 1, config["num_epochs"])
        )
        if epoch < config["global"]["epoch_first"]:
            record_list = poison_linear_record(
                linear_model, poison_eval_loader, split_criterion
            )
            if epoch % config["global"]["t"] == 0 and epoch != 0:
                choice_num += config["global"]["n"]

            logger.info("Mining clean data by class-aware loss-guided split...")
            split_idx = class_aware_loss_guided_split(record_list, indice, all_data_info, choice_num, logger)
            xdata = MixMatchDataset(poison_train_data, split_idx, labeled=True)
            udata = MixMatchDataset(poison_train_data, split_idx, labeled=False)
        elif epoch < config["global"]["epoch_second"]:       
            record_list = poison_linear_record(
                linear_model, poison_eval_loader, split_criterion
            )
            logger.info("Mining clean data by class-agnostic loss-guided split...")
            split_idx = class_agnostic_loss_guided_split(record_list, config["global"]["epsilon"], logger)

            xdata = MixMatchDataset(poison_train_data, split_idx, labeled=True)
            udata = MixMatchDataset(poison_train_data, split_idx, labeled=False)
        elif epoch < config["global"]["epoch_third"]:       
            record_list = poison_linear_record(
                linear_model, poison_eval_loader, split_criterion
            )
            meta_virtual_model = deepcopy(linear_model)
            meta_optimizer_config = config["meta"]["optimizer"]
            param_meta = [
                            {'params': meta_virtual_model.backbone.layer3.parameters()},
                            {'params': meta_virtual_model.backbone.layer4.parameters()},
                            {'params': meta_virtual_model.linear.parameters()}
                        ]
            if "Adam" in meta_optimizer_config:
                meta_optimizer = torch.optim.Adam(param_meta, **meta_optimizer_config["Adam"])
            elif "SGD" in meta_optimizer_config:
                meta_optimizer = torch.optim.SGD(param_meta, **meta_optimizer_config["SGD"])
            meta_criterion = get_criterion(config["meta"]["criterion"])
            meta_criterion = meta_criterion.cuda(gpu)
            for _ in range(config["meta"]["epoch"]):
                train_the_virtual_model(
                                        meta_virtual_model=meta_virtual_model, 
                                        poison_train_loader=poison_train_loader, 
                                        meta_optimizer=meta_optimizer,
                                        meta_criterion=meta_criterion,
                                        gpu=gpu
                                        )      
            meta_record_list = poison_linear_record(
                meta_virtual_model, poison_eval_loader, split_criterion
            )

            logger.info("Mining clean data by meta-split...")
            split_idx = meta_split(record_list, meta_record_list, config["global"]["epsilon"], logger)

            xdata = MixMatchDataset(poison_train_data, split_idx, labeled=True)
            udata = MixMatchDataset(poison_train_data, split_idx, labeled=False)  


        xloader = get_loader(
            xdata, config["semi"]["loader"], shuffle=True, drop_last=True
        )
        uloader = get_loader(
            udata, config["semi"]["loader"], shuffle=True, drop_last=True
        )
        logger.info("MixMatch training...")
        poison_train_result = mixmatch_train(
            linear_model,
            xloader,
            uloader,
            semi_criterion,
            optimizer,
            epoch,
            logger,
            **config["semi"]["mixmatch"]
        )

        logger.info("Test model on clean data...")
        clean_test_result = linear_test(
            linear_model, clean_test_loader, criterion, logger
        )

        logger.info("Test model on poison data...")
        poison_test_result = linear_test(
            linear_model, poison_test_loader, criterion, logger
        )

        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
            )

        # Save result and checkpoint.
        if not args.distributed or (args.distributed and gpu == 0):
            result = {
                "poison_train": poison_train_result,
                "clean_test": clean_test_result,
                "poison_test": poison_test_result,
            }
            result2csv(result, args.log_dir)

            saved_dict = {
                "epoch": epoch,
                "result": result,
                "model_state_dict": linear_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
            }
            if scheduler is not None:
                saved_dict["scheduler_state_dict"] = scheduler.state_dict()

            is_best = False
            if clean_test_result["acc"] > best_acc:
                is_best = True
                best_acc = clean_test_result["acc"]
                best_epoch = epoch + 1
            logger.info(
                "Best test accuaracy {} in epoch {}".format(best_acc, best_epoch)
            )
            if is_best:
                ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
                torch.save(saved_dict, ckpt_path)
                logger.info("Save the best model to {}".format(ckpt_path))
            ckpt_path = os.path.join(args.ckpt_dir, "latest_model.pt")
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the latest model to {}".format(ckpt_path))


def class_aware_loss_guided_split(record_list, has_indice, all_data_info, choice_num, logger):
    """Adaptively split the poisoned dataset by class-aware loss-guided split.
    """
    keys = [r.name for r in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_idx = np.zeros(len(loss))

    total_indice = has_indice.tolist()
    for k, v in all_data_info.items():
        v = np.array(v)
        loss_class = loss[v]
        indice_class = loss_class.argsort()[: choice_num]
        indice = v[indice_class]
        total_indice += indice.tolist()
    total_indice = np.array(total_indice)
    clean_pool_idx[total_indice] = 1

    logger.info(
        "{}/{} poisoned samples in clean data pool".format(poison[total_indice].sum(), clean_pool_idx.sum())
    )
    return clean_pool_idx


def class_agnostic_loss_guided_split(record_list, ratio, logger):
    """Adaptively split the poisoned dataset by class-agnostic loss-guided split.
    """
    keys = [r.name for r in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_idx = np.zeros(len(loss))

    indice = loss.argsort()[: int(len(loss) * ratio)]
    logger.info(
        "{}/{} poisoned samples in clean data pool".format(poison[indice].sum(), len(indice))
    )
    clean_pool_idx[indice] = 1

    return clean_pool_idx


def meta_split(record_list, meta_record_list, ratio, logger):
    """Adaptively split the poisoned dataset by meta-split.
    """
    keys = [r.name for r in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    meta_loss = meta_record_list[keys.index("loss")].data.numpy()
    poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_idx = np.zeros(len(loss))
    loss = loss - meta_loss

    indice = loss.argsort()[: int(len(loss) * ratio)]
    logger.info(
        "{}/{} poisoned samples in clean data pool".format(poison[indice].sum(), len(indice))
    )
    clean_pool_idx[indice] = 1

    return clean_pool_idx


def train_the_virtual_model(meta_virtual_model, poison_train_loader, meta_optimizer, meta_criterion, gpu):
    """Train the virtual model in meta-split.
    """
    meta_virtual_model.train()
    for batch_idx, batch in enumerate(poison_train_loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)

        meta_optimizer.zero_grad()
        output = meta_virtual_model(data)
        meta_criterion.reduction = "mean"
        loss = meta_criterion(output, target)
        
        loss.backward()
        meta_optimizer.step()


if __name__ == "__main__":
    main()
