# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger


from GF3D_org.Group_Free_3D.models2.detector import GroupFreeDetector

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    parser.add_argument("--run", default=1, type=int) # experiment number

    return parser


def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion,
    dataset_config,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    logger = Logger(args.checkpoint_dir)

    # PLOT ACC GRAPH      
    loss_vals =  []
    val_loss_vals =  []
    epochs = []
    val_epochs = []
    mAPs_25 = []
    mAPs_50 = []
    # l_rates = []
    # dlr_rates = []

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)

        aps, epoch_loss = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            dataloaders["train"],
            logger,
        )

        # Loss values
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        metrics = aps.compute_metrics()
        metric_str = aps.metrics_to_str(metrics, per_class=False)
        metrics_dict = aps.metrics_to_dict(metrics)
        curr_iter = epoch * len(dataloaders["train"])
        if is_primary():
            print("==" * 10)
            print(f"Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
            print("==" * 10)
            logger.log_scalars(metrics_dict, curr_iter, prefix="Train/")

        if (
            epoch > 0
            and args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
            )

        if epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1):
            ap_calculator, val_loss = evaluate(
                args,
                epoch,
                model,
                criterion,
                dataset_config,
                dataloaders["test"],
                logger,
                curr_iter,
            )
            # Val Loss
            val_loss_vals.append(sum(val_loss)/len(val_loss))
            val_epochs.append(epoch)

            metrics = ap_calculator.compute_metrics()
            ap25 = metrics[0.25]["mAP"]
            ##
            ap50 = metrics[0.5]["mAP"]
            mAPs_25.append(ap25*100)
            mAPs_50.append(ap50*100)
            ##
            #Plot
            plt.figure(figsize=(8, 6))
            plt.plot(val_epochs, mAPs_25, 'bo-', val_epochs, mAPs_50,'ro-')
            plt.legend(['mAP@25', 'mAP@50'])
            plt.title('mAP vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('mAP')
            plt.savefig(os.path.join('/l/users/aidana.nurakhmetova/thesis/3DETR_2/3detr/acc_graphs', f'acc_graph_{epoch}_run{args.run}.jpg'))
            plt.show()
            ####
            metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ap_calculator.metrics_to_dict(metrics)
            if is_primary():
                print("==" * 10)
                print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)
                logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")

            if is_primary() and (
                len(best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25}"
                )
    # PLOT
    plt.figure(figsize=(8, 6))
    plt.plot(val_epochs, mAPs_25, 'bo-', val_epochs, mAPs_50,'ro-')
    plt.legend(['mAP@25', 'mAP@50'])
    plt.title('mAP vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.savefig(os.path.join('/l/users/aidana.nurakhmetova/thesis/3DETR_2/3detr/acc_graphs', f'acc_graph_run{args.run}.jpg'))
    plt.show()

    # always evaluate last checkpoint
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ap_calculator, val_loss = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Evaluate Final [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
        print("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ap_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)


def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    datasets, dataset_config = build_dataset(args)
    model, _ = build_model(args, dataset_config)
    model = model.cuda(local_rank)
    model_no_ddp = model

     ## CREARE GF3D MODEL INSTANCE
    num_input_channel = int(args.use_color) * 3
    mean_size_arr = [
                    [0.76966726, 0.81160211, 0.92573741],
                    [1.876858,  1.84255952, 1.19315654],
                    [0.61327999, 0.61486087, 0.71827014],
                    [1.39550063, 1.51215451, 0.83443565],
                    [0.97949596, 1.06751485, 0.63296875],
                    [0.53166301, 0.59555772, 1.75001483],
                    [0.96247056, 0.72462326, 1.14818682],
                    [0.83221924, 1.04909355, 1.68756634],
                    [0.21132214, 0.4206159,  0.53728459],
                    [1.44400728, 1.89708334, 0.26985747],
                    [1.02942616, 1.40407966, 0.87554322],
                    [1.37664116, 0.65521793, 1.68131292],
                    [0.66508189, 0.71111926, 1.29885307],
                    [0.41999174, 0.37906947, 1.75139715],
                    [0.59359559, 0.59124924, 0.73919014],
                    [0.50867595, 0.50656087, 0.30136236],
                    [1.15115265, 1.0546296,  0.49706794],
                    [0.47535286, 0.49249493, 0.58021168]]
    mean_size_arr = np.array(mean_size_arr)

    from GF3D_org.Group_Free_3D.sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()

    gf_model = GroupFreeDetector(num_class=DATASET_CONFIG.num_class,  #dataset_config.num_semcls
                              num_heading_bin=DATASET_CONFIG.num_heading_bin, #dataset_config.num_angle_bin
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,  # dataset_config.num_semcls, num_size_cluster == num_semcls
                              mean_size_arr= DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=1,
                              bn_momentum=0.1,
                              sync_bn=False,
                              num_proposal=256, #
                              sampling='kps',
                              dropout=0.1,
                              activation='relu',
                              nhead=8,
                              num_decoder_layers=6, # 12
                              dim_feedforward=2048,
                              size_cls_agnostic=True,
                            #   dataset='sunrgbd',
                              self_position_embedding='loc_learned',
                              cross_position_embedding='xyz_learned',
                              )

    # print("GF3D backbone: ", group_free_model.backbone_net)
    weight_dir = '/l/users/aidana.nurakhmetova/thesis/3DETR_2/3detr/sunrgbd_l6o256_cls_agnostic.pth'
    # load_pretrained_group_free(weight_dir,group_free_model)
    checkpoint = torch.load(weight_dir, map_location='cpu')
    # gf_model.load_state_dict(checkpoint['model'], ) #strict=False
    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    gf_model.load_state_dict(checkpoint['model'] )

    for param in gf_model.parameters():
        if param.requires_grad:
            param.requires_grad = False

    state_dict2 = model.state_dict()

    for k1 in state_dict.keys():
        for k2 in state_dict2.keys(): 
            if k1 == k2:
            # if k.startswith("backbone_net."):
                # print("value ", k1, state_dict[k1].shape, state_dict2[k2].shape)
                if state_dict[k1].shape == state_dict2[k2].shape:
                    # CHECK THE MODIFICATION
                    #print("previous value: ", k2, state_dict2[k2].shape)
                    # print(state_dict2[k2])
                    # CHANGE THE 3DETR WEIGHTS TO GF3D WEIGHTS
                    state_dict2[k2] = state_dict[k1]
                    #print("modified value ", k2, state_dict2[k2].shape)
                    # print(state_dict2[k2])

    for name1, param1 in gf_model.named_parameters():
        for name2, param2 in model.named_parameters():
            if name1 == name2:
                # print("name ", name1, param1.shape, param2.shape)
                if param1.shape == param2.shape:
                    with torch.no_grad():
                        param2.copy_(param1)

    model.load_state_dict(state_dict2)
    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    criterion = build_criterion(args, dataset_config)
    criterion = criterion.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        criterion = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            criterion,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
