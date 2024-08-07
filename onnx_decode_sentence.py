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
from decode import get_metrics, print_metrics, case_id, punct_id
from decode_sentence import encode_sentences, decode_sentences
from onnx_decode import OnnxModel
import onnx
from onnx import numpy_helper, helper

##### usage
## python3 onnx_decode_sentence.py --text_file ./test.txt --exp_dir ../output/ --bpe_model ../bpe_model/bpe.model

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The sentences in text_file are to be decoded")
    parser.add_argument("--model_filename",
                        type=str,
                        default="",
                        required=True,
                        help="The onnx model file path")
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

    return parser

@torch.no_grad()
def main():
    parser = get_parser()

    args = parser.parse_args()
    # args.exp_dir = Path(args.exp_dir)
    log_dir = os.path.dirname(args.model_filename)
    params = get_params()
    params.update(vars(args))

    random.seed(42)
    torch.manual_seed(42)

    setup_logger(f"{log_dir}/log-onnx-decode-sentence")
    logging.info("Decoding started")

    device = torch.device("cpu")
    # rank = 0 # hardcode 0 to use single GPU firstly
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    logging.info(params)

    model = OnnxModel(model_filename = args.model_filename)

    token_ids, valid_ids, label_lens, label_masks = encode_sentences(params.text_file, sp, device)

    active_case_logits, active_punct_logits, _ = model.run_model(token_ids, valid_ids, label_lens)

    case_pred = torch.argmax(active_case_logits, dim=1)
    punct_pred = torch.argmax(active_punct_logits, dim=1)

    logging.info(
            f"Result --> "
            f"case pred:{case_pred}, "
            f"punct pred:{punct_pred}"
        )

    decode_sentences(params.text_file, case_pred, punct_pred)


if __name__ == "__main__":
    main()