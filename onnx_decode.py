import argparse
import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from utils import (AttributeDict, setup_logger)
import sentencepiece as spm
from data_module import DataModule, sort_batch
from tqdm import tqdm
from typing import List, Tuple
from train import get_params
from decode import get_metrics, print_metrics, case_id, punct_id
import torch.nn.functional as F
import random
import sys

import onnx
import onnxruntime as ort
from onnx import numpy_helper, helper

############ Usage:
### python3 onnx_decode.py --model_filename ../output/model.onnx --data_dir ../data/ --bpe_model ../bpe_model/


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_filename",
        type=str,
        default="",
        required=True,
        help="The onnx model file path",
    )
    parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		required=True,
		help="The input data dir. Should include text file - words.txt and label file - labels.txt"
    )
    parser.add_argument(
    	"--bpe_model",
        default=None,
        type=str,
        required=True,
        help="The bpe model path"
	)
    parser.add_argument(
		"--max_seq_length",
		default=200,
		type=int,
		# required=True,
		help="The sequence length of one sample after SentencePiece tokenization"
    )
    parser.add_argument(
    	"--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
   	)
    parser.add_argument(
    	"--batch_size",
        default=256,
        type=int,
        # required=True,
        help="Batch size for decoding"
    )

    return parser

class OnnxModel:
	def __init__(self, model_filename: str):
		session_opts = ort.SessionOptions()
		session_opts.inter_op_num_threads = 1
		session_opts.intra_op_num_threads = 1
		# session_opts.use_deterministic_compute = 1

		self.session_opts = session_opts

		self.init_model(model_filename)

	def init_model(self, model_filename: str):
		self.model = ort.InferenceSession(
			model_filename,
			sess_options = self.session_opts,
			providers = ["CPUExecutionProvider"],
		)

	def run_model(
		self, 
		token_ids: torch.Tensor, 
		valid_ids: torch.Tensor, 
		label_lens: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:

		out = self.model.run(
			[
				self.model.get_outputs()[0].name,
			 	self.model.get_outputs()[1].name,
			 	self.model.get_outputs()[2].name,
			 	# self.model.get_outputs()[3].name
			],
			{
				self.model.get_inputs()[0].name: token_ids.cpu().numpy(),
				self.model.get_inputs()[1].name: valid_ids.cpu().numpy(),
				self.model.get_inputs()[2].name: label_lens.cpu().numpy(),
			},
		)

		return torch.from_numpy(out[0]), torch.from_numpy(out[1]), torch.from_numpy(out[2]) #, torch.from_numpy(out[3])

@torch.no_grad()
def main():
	parser = get_parser()

	args = parser.parse_args()
	# args.exp_dir = Path(args.exp_dir)
	# args.model_filename = Path(args.model_filename)
	log_dir = os.path.dirname(args.model_filename)
	params = get_params()
	params.update(vars(args))

	random.seed(42)
	torch.manual_seed(42)

	setup_logger(f"{log_dir}/log-onnx-decode")
	logging.info("Decoding started")

	device = torch.device("cpu")
	# rank = 0 # hardcode 0 to use single GPU firstly
	# if torch.cuda.is_available():
	#     device = torch.device("cuda", rank)
	logging.info(f"Device: {device}")

	sp = spm.SentencePieceProcessor()
	sp.load(args.bpe_model)

	logging.info(vars(args))

	model = OnnxModel(model_filename = args.model_filename)

	data_module = DataModule(args, sp)
	decode_dl, test_file = data_module.test_dataloader()
	logging.info(f"test_file:{test_file}, len(decode_dl):{len(decode_dl)}")
    
	for batch_idx, batch in enumerate(tqdm(decode_dl)):
		batch = tuple(t.to(device, dtype=torch.int32) for t in batch)
		token_ids, label_ids, valid_ids, label_lens, label_masks = batch

		active_case_logits, active_punct_logits, mask = model.run_model(token_ids, valid_ids, label_lens)

		label_lens, indx = torch.sort(label_lens, dim=0, descending=True, stable=True)
		label_ids = label_ids[indx]

		case_pred = torch.argmax(active_case_logits, dim=1)
		punct_pred = torch.argmax(active_punct_logits, dim=1)

		label_ids = label_ids[:, :, :mask.shape[1]]
		active_case_labels = label_ids[:, 0, :][mask]
		active_punct_labels = label_ids[:, 1, :][mask]

		precision_case, recall_case, f_scores_case, overall_case = get_metrics(case_pred.detach().cpu().numpy(), active_case_labels.detach().cpu().numpy())
		precision_punct, recall_punct, f_scores_punct, overall_punct = get_metrics(punct_pred.detach().cpu().numpy(), active_punct_labels.detach().cpu().numpy())

		# logging.info("\nCase metrics:\n----------------------------------------------------------------------------------------")
		# print_metrics(logging, precision_case, recall_case, f_scores_case, overall_case, case_id)
		logging.info("\nPunct metrics:\n=======================================================================================")
		print_metrics(logging, precision_punct, recall_punct, f_scores_punct, overall_punct, punct_id)


if __name__ == "__main__":
	main()