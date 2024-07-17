import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from torch import Tensor
import random
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List
import sentencepiece as spm
from model import Model,Model_new,Model_test
from data_module import DataModule, sort_batch
import torch.distributed as dist
from datetime import datetime
from datetime import timedelta
from torch.optim import Optimizer
from utils import (
    AttributeDict,
    setup_logger,
    str2bool,
    setup_dist,
    cleanup_dist,
    LRScheduler,
    Eden,
)

##### usage
# export NCCL_P2P_LEVEL=NVL | export NCCL_DEBUG=INFO | export CUDA_VISIBLE_DEVICES="0,1,2,3"
## python3 train.py --world-size 4 --data_dir ../data/ --exp_dir ../output/ --bpe_model ../bpe_model/bpe.model --epochs 30

### fine-tune:
"""
python3 train.py \
	--world-size 4 \
	--do_finetune True \
	--finetune_ckpt ../output_lr002_384_BS64_3conv_1fc_Add\&Norm_WeightsInitialize_30epoch/epoch-15.pt \
	--data_dir ../finetune_data/ \
	--exp_dir ../output/ \
	--bpe_model ../bpe_model/bpe.model \
	--base-lr 0.0001 \
	--epochs 20
"""

def add_finetune_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--do_finetune",
        type=str2bool,
        default=False,
        help="Whether to fine-tune.",
    )
    parser.add_argument(
        "--use_mux",
        type=str2bool,
        default=False,
        help="""
        Whether to adapt. If true, we will mix 5% of the new data
        with 95% of the original data to fine-tune.
        """,
    )

    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default=None,
        help="Fine-tuning from which checkpoint (a path to a .pt file)",
    )

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--world-size",
						type=int,
						default=1,
						help="Number of GPUs for DDP training.",)

    parser.add_argument("--master-port",
						type=int,
						default=12354,
						help="Master port to use for DDP training.",)

    parser.add_argument("--tensorboard",
						type=str2bool,
						default=True,
						help="Should various information be logged in tensorboard.",)

    parser.add_argument("--start-epoch",
						type=int,
						default=1,
						help="""Resume training from this epoch. It should be positive.
						If larger than 1, it will load checkpoint from
						exp-dir/epoch-{start_epoch-1}.pt
						""",)

    parser.add_argument("--start-batch",
						type=int,
						default=0,
						help="""If positive, --start-epoch is ignored and
						it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
						""",)

    parser.add_argument("--data_dir",
                    	default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should include text file - words.txt and label file - labels.txt")
    parser.add_argument("--exp_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--bpe_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The bpe model path")
    parser.add_argument("--max_seq_length",
                        default=200,
                        type=int,
                        # required=True,
                        help="The sequence length of one sample after SentencePiece tokenization")
    parser.add_argument("--batch_size",
                        default=256, # 64 
                        type=int,
                        # required=True,
                        help="Batch size for training")
    # parser.add_argument("--learning_rate",
    #                     default=0.001, # 5e-5
    #                     type=float,
    #                     # required=True,
    #                     help="The initial learning rate for Adam")
    parser.add_argument("--base-lr", 
    					type=float, 
    					default=0.002, 
    					help="The base learning rate.")

    parser.add_argument("--lr-batches",
						type=float,
						default=7500, #7500
						help="""Number of steps that affects how rapidly the learning rate
						decreases. We suggest not to change this.""",)

    parser.add_argument("--lr-epochs",
						type=float,
						default=3.5,
						help="""Number of epochs that affects how rapidly the learning rate decreases.
						""",)
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        # required=True,
                        help="Total number of training epochs to perform")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        # required=True,
                        help="Proportion of training to perform linear learning rate warmup, e.g. 0.1=10\% of training")
    parser.add_argument("--weight_decay",
                        default=2.5e-5,
                        type=float,
                        # required=True,
                        help="Weight decay if we apply some")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        # required=True,
                        help="random seed for initialization")

    add_finetune_arguments(parser)

    return parser

def get_params() -> AttributeDict:
	params = AttributeDict(
		{
			"vocab_size": 5000,
			"embedding_dim": 100, # 100
			"sequence_size": 200,
			"hidden_size1": 384,
			"hidden_size2": 384,
			"out_size_case": 4,
			"out_size_punct": 4,
			"dropout": 0.5,
			"gamma": 0.5, # StepLR
			"factor": 0.8,  # ReduceLROnPlateau
			"patience": 2,  # ReduceLROnPlateau
			"log_interval": 100, # 100, for ~1000 trian dl length # 200 ~ big data
			"valid_interval": 300, #300, for ~1000 trian dl length # 800 ~ big data
			"batch_idx_train": 0,
			"save_every_n": 200, # 200, for ~1000 trian dl length # 300 ~ big data
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "case_loss": -1,
            "punct_loss": -1,
		}
	)

	return params

def get_model(params) -> nn.Module:

	model = Model_new(params)

	return model

def load_model_params(
    ckpt: str, model: nn.Module, strict: bool = True
):
    """Load model params from checkpoint

    Args:
        ckpt (str): Path to the checkpoint
        model (nn.Module): model to be loaded

    """
    logging.info(f"Loading checkpoint from {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")

    if next(iter(checkpoint["model"])).startswith("module."):
        logging.info("Loading checkpoint saved by DDP")

        dst_state_dict = model.state_dict()
        src_state_dict = checkpoint["model"]
        for key in dst_state_dict.keys():
            src_key = "{}.{}".format("module", key)
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        model.load_state_dict(dst_state_dict, strict=strict)
    else:
        model.load_state_dict(checkpoint["model"], strict=strict)

    return None

def save_checkpoint(
	params: AttributeDict,
	model: Union[nn.Module, DDP],
	optimizer: Optional[torch.optim.Optimizer] = None,
	scheduler: Optional[LRScheduler] = None,
	# sampler: Optional[Sampler] = None,
	rank: int = 0,
) -> None:
	if rank != 0:
		return

	filename = params.exp_dir + f"/epoch-{params.cur_epoch}.pt"

	logging.info(f"Saving checkpoint to {filename}")

	if isinstance(model, DDP):
		model = model.module

	checkpoint = {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict() if optimizer is not None else None,
		"scheduler": scheduler.state_dict() if scheduler is not None else None,
		# "sampler": sampler.state_dict() if sampler is not None else None,
	}

	torch.save(checkpoint, filename)

	if params.best_train_epoch == params.cur_epoch:
		best_train_filename = params.exp_dir + "/best-train-loss.pt"
		logging.info(f"Saving best-train-loss checkpoint to {best_train_filename}")
		torch.save(checkpoint, best_train_filename)

	# 	# copyfile(src=filename, dst=best_train_filename)

	if params.best_valid_epoch == params.cur_epoch:
		best_valid_filename = params.exp_dir + "/best-valid-loss.pt"
		logging.info(f"Saving best-valid-loss checkpoint to {best_valid_filename}")
		torch.save(checkpoint, best_valid_filename)
	# 	# copyfile(src=filename, dst=best_valid_filename)

def save_checkpoint_with_global_batch_idx(
    global_batch_idx: int,
    model: Union[nn.Module, DDP],
    params: AttributeDict,	
    # optimizer: Optional[torch.optim.Optimizer] = None,
    # scheduler: Optional[LRScheduler] = None,
    # sampler: Optional[Sampler] = None,
    rank: int = 0,
) -> None:

	if rank != 0:
		return

	filename = params.exp_dir + f"/checkpoint-{global_batch_idx}.pt"

	logging.info(f"Saving checkpoint to {filename}")

	if isinstance(model, DDP):
		model = model.module

	checkpoint = {
		"model": model.state_dict(),
		# "optimizer": optimizer.state_dict() if optimizer is not None else None,
		# "scheduler": scheduler.state_dict() if scheduler is not None else None,
		# "sampler": sampler.state_dict() if sampler is not None else None,
	}

	torch.save(checkpoint, filename)

def compute_loss(
	model: Union[nn.Module, DDP],
	batch: dict,
	device: torch.device,
	params: AttributeDict,
):
	batch = tuple(t.to(device) for t in batch)
	token_ids, label_ids, valid_ids, label_lens, label_masks = batch
	
	case_loss, punct_loss = model(token_ids, valid_ids=valid_ids, label_lens=label_lens, label_masks=label_masks, labels=label_ids)
	# loss = 0.4*case_loss + 0.6*punct_loss	
	loss = case_loss + 0.7*punct_loss

	# for log
	params.case_loss = case_loss
	params.punct_loss = punct_loss

	return loss

def compute_validation_loss(
	params: AttributeDict,
	model: Union[nn.Module, DDP],
	valid_dl: torch.utils.data.DataLoader,
	device: torch.device
) -> Tensor:
	model.eval()

	tot_loss = 0

	with torch.no_grad():
		for batch_idx, batch in enumerate(valid_dl):
			loss = compute_loss(model, batch, device, params)
			tot_loss = tot_loss + loss.detach().cpu().item()
			# tot_loss = tot_loss + params.punct_loss.detach().cpu().item()

	if tot_loss < params.best_valid_loss:
		params.best_valid_loss = tot_loss
		params.best_valid_epoch = params.cur_epoch

	return tot_loss / len(valid_dl)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

def run(rank, world_size, args):
	params = get_params()
	params.update(vars(args))

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	if world_size > 1:
		setup_dist(rank, world_size, params.master_port)

	setup_logger(f"{params.exp_dir}/log-train")

	device = torch.device("cpu")
	# rank = 3 # hardcode 0 to use single GPU firstly
	if torch.cuda.is_available():
		device = torch.device("cuda", rank)
	logging.info(f"Device: {device}")

	# add <SOS>, <EOS>, <PAD> token?
	sp = spm.SentencePieceProcessor()
	sp.load(args.bpe_model)

	params.vocab_size = sp.get_piece_size()

	logging.info(params)

	model = get_model(params)
	logging.info(model)

	num_param = sum([p.numel() for p in model.parameters()])
	logging.info(f"Number of model parameters: {num_param}")

    # assert params.start_epoch > 0, params.start_epoch
    # checkpoints = load_checkpoint_if_available(
    #     params=params, model=model
    # )
	if params.do_finetune:
		checkpoints = load_model_params(
			ckpt=params.finetune_ckpt, model=model
		)
		params.log_interval = 5
		params.valid_interval = 15
		params.save_every_n = 25

	#### for train finetune data only
	# params.log_interval = 10
	# params.valid_interval = 60
	# params.save_every_n = 100

	if args.tensorboard and rank == 0:
		tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
	else:
		tb_writer = None

	model.to(device)

	if world_size > 1:
		logging.info("Using DDP")
		model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	model.apply(initialize_weights)

	# TODO: try AdamW,https://github.com/Unbabel/caption/blob/shared-task/models/punct_predictor.py
	optimizer = torch.optim.Adam(model.parameters(), lr=params.base_lr, weight_decay=params.weight_decay)
	# print(f"------------------------------------------>\nmodel.parameters():\n")
	# for param in model.parameters():
	# 	print(type(param.data), param.size())
	# add warm-ups and scheduler
	# scheduler = Eden2(optimizer, params.lr_batches)
	# scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
	# TODO: try ReduceLROnPlateau/CosinAnnealingLR
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=params.gamma)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
											optimizer=optimizer,
											factor=params.factor, 
											patience=params.patience,
											threshold=0.0005,
											threshold_mode='abs',
											verbose=True,
											)

	data_module = DataModule(args, sp)
	train_dl = data_module.train_dataloader()
	# train_dl = data_module.valid_dataloader()
	valid_dl = data_module.valid_dataloader()
	# valid_dl, test_file = data_module.test_dataloader()

	# print("debug dataloader")
	logging.info(f"len(train_dl):{len(train_dl)}")
	logging.info(f"len(valid_dl):{len(valid_dl)}")

	# it = iter(train_dl)

	# token_ids, label_ids, valid_ids, label_lens, label_masks = next(it)
	# print(f"type(token_ids):{type(token_ids)}")
	# print(f"token_ids:{token_ids}")
	

	logging.info("Training started")

	# add start_epoch
	for epoch in range(args.epochs + 1):
		params.cur_epoch = epoch

		if world_size > 1:
			train_dl.sampler.set_epoch(epoch)

		if tb_writer is not None:
			tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

		model.train()

		tot_loss = 0

		for batch_idx, batch in enumerate(train_dl):
			params.batch_idx_train += 1
			# batch = tuple(t.to(device) for t in batch)
			# token_ids, label_ids, valid_ids, label_lens, label_masks = batch
			# # print("token_ids:")
			# # print(f"{token_ids}")

			# case_loss, punct_loss = model(token_ids, valid_ids=valid_ids, label_lens=label_lens, label_masks=label_masks, labels=label_ids)
			# # if params.batch_idx_train > 2:
			# # 	loss = 0.4*case_loss + 0.6*punct_loss + 0.4*penalty
			# # else:
			# # 	loss = 0.4*case_loss + 0.6*punct_loss
			# loss = 0.4*case_loss + 0.6*punct_loss
			# # logging.info(f"case_loss:{case_loss}, punct_loss:{punct_loss},penalty:{penalty}")

			loss = compute_loss(model, batch, device, params)

			loss.backward()
			# scheduler.step_batch(params.batch_idx_train)
			optimizer.step()
			optimizer.zero_grad()
			tot_loss = tot_loss + loss.detach().cpu().item()

			if batch_idx % params.log_interval == 0:
				# cur_lr = max(scheduler.get_last_lr()) # for StepLR
				# cur_lr = scheduler.get_last_lr()
				cur_lr = optimizer.param_groups[0]['lr'] # for ReduceLROnPlateau
				logging.info(
					f"Epoch {params.cur_epoch}, "
					f"batch {batch_idx}, "
					f"lr: [{cur_lr:.2e}], "
					f"loss[{loss.detach().cpu().item()}], "
					f"case_loss[{params.case_loss}], "
					f"punct_loss[{params.punct_loss}], "
					# f"penalty[{penalty}], "
					# f"alpha[{alpha.detach().cpu().item()}], "
					# f"beta[{beta.detach().cpu().item()}]"
				)

				if tb_writer is not None:
					tb_writer.add_scalar(
						"train/learning_rate", cur_lr, params.batch_idx_train
					)
					tb_writer.add_scalar(
						"train/current_loss", loss.detach().cpu().item(), params.batch_idx_train
					)
					tb_writer.add_scalar(
						"train/tot_loss", tot_loss, params.batch_idx_train
					)

			if params.batch_idx_train % params.save_every_n == 0:
				save_checkpoint_with_global_batch_idx(
	                global_batch_idx=params.batch_idx_train,
	                model=model,
	                params=params,
					# optimizer=optimizer,
					# scheduler=scheduler,
					# sampler=train_dl.sampler,
	                rank=rank,
				)

			if (batch_idx+1) % params.valid_interval == 0:
				logging.info("start validation")
				valid_loss = compute_validation_loss(
						params=params,
						model=model, 
						valid_dl=valid_dl,
						device=device,
						)
				model.train()

				# only for ReduceLROnPlateau, should be called after validate()
				scheduler.step(valid_loss)

				logging.info(f"Epoch {params.cur_epoch}, validation loss[{valid_loss}]")

				if tb_writer is not None:
					tb_writer.add_scalar(
						"train/valid_loss", valid_loss, params.batch_idx_train
					)

		# scheduler.step() # for StepLR

		if tot_loss < params.best_train_loss:
			params.best_train_loss = params.best_train_loss
			params.best_train_epoch = params.cur_epoch

		save_checkpoint(
			params=params,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			# sampler=train_dl.sampler,
			rank=rank,
		)

	logging.info("Done!")

	if world_size > 1:
		torch.distributed.barrier()
		cleanup_dist()

def main():
    parser = get_parser()
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
	main()