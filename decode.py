import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor
from pathlib import Path
import random
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union
import sentencepiece as spm
from model import Model
from data_module import DataModule, sort_batch
import torch.distributed as dist
from datetime import datetime
import torch.nn.functional as F
from train import get_model, get_params
from utils import (AttributeDict, setup_logger)
from tqdm import tqdm

import onnxruntime as ort
import numpy as np

##### usage
## python3 decode.py --data_dir ../data/ --exp_dir ../output/ --bpe_model ../bpe_model/bpe.model --batch 1000

###### !!! keep align with process_data.py
punct_id = {0:"NO_PUNCT",
             1:"COMMA",
             2:"PERIOD",
             3:"QUESTION",
            }
case_id = {0:"LOWER",
            1:"UPPER",
            2:"CAP",
            3:"MIX_CASE",
            }

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should include text file - words.txt and label file - labels.txt")
    parser.add_argument("--exp_dir",
                    	default=None,
                        type=str,
                        required=True,
                        help="The experiment dir contains .pt")
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
                        default=1024,
                        type=int,
                        # required=True,
                        help="Batch size for decoding")
    parser.add_argument("--world-size",
                        type=int,
                        default=1,
                        help="Number of GPUs for DDP training.",)
    parser.add_argument("--epoch",
                        default=-1,
                        type=int,
                        # required=True,
                        help="The epoch pt used for decoding")
    parser.add_argument("--batch",
                        default=-1,
                        type=int,
                        # required=True,
                        help="The batch pt used for decoding")

    return parser 

def inc(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1

def get_metrics(output, target):
    assert len(output) == len(target), f"output len:{output} != target len:{target}"

    true_predicted = {}
    all_predicted = {}
    all_expected = {}

    for i in range(len(output)):

        inc(all_expected, target[i])
        inc(all_predicted, output[i])
        if target[i] == output[i]:
            inc(true_predicted, output[i])

    # print(f"all_predicted:{all_predicted}")
    # print(f"all_expected:{all_expected}")
    # print(f"true_predicted:{true_predicted}")

    precision = {k: (true_predicted[k] if k in true_predicted else 0) / all_predicted[k] for k in all_predicted.keys()}
    recall = {k: (true_predicted[k] if k in true_predicted else 0) / all_expected[k] for k in all_expected.keys()}

    f_scores = {
        k: None if precision[k] == 0 else (0 if recall[k] == 0 else (2*precision[k]*recall[k]/(precision[k]+recall[k])))
        for k in precision
    }

    overall_true_predicted = 0
    overall_all_predicted = 0
    overall_all_expected = 0
    for k in all_expected.keys():
        if k > 0:
            overall_true_predicted += (true_predicted[k] if k in true_predicted else 0)
            overall_all_predicted += (all_predicted[k] if k in all_predicted else 0)
            overall_all_expected += all_expected[k]
    overall_precision = (overall_true_predicted / overall_all_predicted if overall_all_predicted > 0 else 0)
    overall_recall = (overall_true_predicted / overall_all_expected if overall_all_expected > 0 else 0)
    overall_f_scores = (2*overall_precision*overall_recall/(overall_precision+overall_recall) if overall_recall > 0 else 0)

    return precision, recall, f_scores, (overall_precision, overall_recall, overall_f_scores)

def print_metrics(logging, precision, recall, f_scores, overall, label_map):
    # print(f"precision:{precision}")

    for k in label_map.keys():
        # print(f"-----------> k:{k} - [{label_map[k]}]")
        logging.info(f"{label_map[k]}: \tPrec [{precision[k]:.3f}], " + 
                    (f"\tRec [{recall[k]:.3f}], " if k in recall else "\tRec [None], ") +
                    (f"\tF1 [{f_scores[k]:.3f}], " if f_scores[k] != None else "\tF1 [None], ") 
                )  
    logging.info(f"Overall: \tPrec [{overall[0]:.3f}], " +
                 f"\tRec [{overall[1]:.3f}], " +
                 f"\tF1 [{overall[2]:.3f}], "
            )

@torch.no_grad()
def main():
    parser = get_parser()

    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))

    random.seed(42)
    torch.manual_seed(42)

    setup_logger(f"{params.exp_dir}/log-decode")
    logging.info("Decoding started")

    device = torch.device("cpu")
    rank = 0 # hardcode 0 to use single GPU firstly
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    print(model)  

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if params.epoch > 0:
        ptfile = f"{params.exp_dir}/epoch-{params.epoch-1}.pt"
    if params.batch > 0:
        ptfile = f"{params.exp_dir}/checkpoint-{params.batch}.pt"
    logging.info(f"Loading checkpoint from {ptfile}")
    checkpoint = torch.load(ptfile, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    checkpoint.pop("model")

    model.to(device)
    model.eval()

    data_module = DataModule(args, sp)
    decode_dl, test_file = data_module.test_dataloader()
    logging.info(f"test_file:{test_file}, len(decode_dl):{len(decode_dl)}")

    for batch_idx, batch in enumerate(tqdm(decode_dl)):
        batch = tuple(t.to(device) for t in batch)
        token_ids, label_ids, valid_ids, label_lens, label_masks = batch

        active_case_logits, active_punct_logits, mask = model(token_ids, valid_ids=valid_ids, label_lens=label_lens) 

        label_lens, indx = torch.sort(label_lens, dim=0, descending=True, stable=True)
        label_ids = label_ids[indx]

        case_pred = torch.argmax(F.log_softmax(active_case_logits, dim=1), dim=1)
        punct_pred = torch.argmax(F.log_softmax(active_punct_logits, dim=1), dim=1)

        label_ids = label_ids[:, :, :mask.shape[1]]
        active_case_labels = label_ids[:, 0, :][mask]
        active_punct_labels = label_ids[:, 1, :][mask]

        precision_case, recall_case, f_scores_case, overall_case = get_metrics(case_pred.detach().cpu().numpy(), active_case_labels.detach().cpu().numpy())
        precision_punct, recall_punct, f_scores_punct, overall_punct = get_metrics(punct_pred.detach().cpu().numpy(), active_punct_labels.detach().cpu().numpy())

        logging.info("\nCase metrics:\n----------------------------------------------------------------------------------------")
        print_metrics(logging, precision_case, recall_case, f_scores_case, overall_case, case_id)
        logging.info("\nPunct metrics:\n=======================================================================================")
        print_metrics(logging, precision_punct, recall_punct, f_scores_punct, overall_punct, punct_id)


if __name__ == "__main__":
    main()