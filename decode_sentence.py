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

##### usage
## python3 decode_sentence.py --text_file ./test.txt --exp_dir ../output/ --bpe_model ../bpe_model/bpe.model --batch 1000

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The sentences in text_file are to be decoded")
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

def encode_sentences(
    text_file, 
    tokenizer:spm, 
    device,
    max_seq_length: int = 200,
):
    text_lines = []
    with open(text_file, "r") as hand:
        for line in hand:
            text_lines.append(line)

    # all examples
    tokens_list = []
    valid_list = []
    label_masks_list = []
    label_len_list = []

    # one example:
    tokens = [tokenizer.piece_to_id('<s>')]
    valid = [1]
    label_masks = [1]
    label_len = 0

    logging.info("Encoding sentences...")
    for il, line in enumerate(tqdm(text_lines)):
        words = line.split()
        for iw, word in enumerate(words):
            word_tokens = tokenizer.encode(word, out_type=int)
            tokens_str = tokenizer.encode(word, out_type=str)
            print(f"tokens_str:{tokens_str}")

            if len(tokens) + len(word_tokens) > max_seq_length - 1:
                tokens.append(tokenizer.piece_to_id('</s>'))
                valid.append(1)
                label_masks.append(1)

                label_len = len(label_masks)

                while len(tokens) < max_seq_length:
                    tokens.append(0)
                    valid.append(0)
                while len(label_masks) < max_seq_length:
                    label_masks.append(0)
                assert len(tokens) == max_seq_length
                assert len(valid) == max_seq_length
                assert len(label_masks) == max_seq_length

                tokens_list.append(tokens)
                valid_list.append(valid)
                label_masks_list.append(label_masks)
                label_len_list.append(label_len)

                tokens = [tokenizer.piece_to_id('<s>')]
                valid = [1]
                label_masks = [1]
                label_len = 0

            tokens.extend(word_tokens)
            for m in range(len(word_tokens)):
                if m == 0:
                    valid.append(1)
                    label_masks.append(1)
                else:
                    valid.append(0)

    if len(tokens) > 0:
        logging.info("Input sentences number is too small to be packed to one sample length(200)")

        tokens.append(tokenizer.piece_to_id('</s>'))
        valid.append(1)
        label_masks.append(1)

        label_len = len(label_masks)

        while len(tokens) < max_seq_length:
            tokens.append(0)
            valid.append(0)
        while len(label_masks) < max_seq_length:
            label_masks.append(0)
        assert len(tokens) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_masks) == max_seq_length

        tokens_list.append(tokens)
        valid_list.append(valid)
        label_masks_list.append(label_masks)
        label_len_list.append(label_len)


    return torch.tensor(tokens_list, dtype=torch.int32, device=device), torch.tensor(valid_list, dtype=torch.int32, device=device), torch.tensor(label_len_list, dtype=torch.int32, device=device), torch.tensor(label_masks_list, dtype=torch.int32, device=device)

def decode_sentences(
    text_file,
    case_pred,
    punct_pred,
):
    text_lines = []
    words_num = 0
    with open(text_file, "r") as hand:
        for line in hand:
            text_lines.append(line.split())   
            words_num += len(line.split())
    print(f"words_num:{words_num}, len(case_pred):{len(case_pred)}")
    assert(words_num == len(case_pred))
    assert(words_num == len(punct_pred))

    words_num = 0
    # print(f"text_lines:{text_lines}")
    for i,l in enumerate(text_lines):
        output = ""
        for j,w in enumerate(l):
            prefix = " " if j != 0 else ""
            output += prefix

            if case_pred[words_num] == 2:
                output += w.title()
            elif case_pred[words_num] == 1:
                output += w.upper()
            else:
                output += w

            suffix = ""
            if punct_pred[words_num] == 1:
                suffix = ","
            elif punct_pred[words_num] == 2:
                suffix = "."
            elif punct_pred[words_num] == 3:
                suffix = "?"

            output += suffix

            words_num += 1

        # print(f"Decode sentence[{i}]:{output}")
        print(f"{output}")

### This func move all 1s to the left of sequence 
### After move, for 1s subsequence, replace the first element 1 and the last element 1 with value 2
def handle_bos_eos(valid_ids):

    sorted_sequences, _ = valid_ids.sort(dim=1, descending=True)

    # Now find the first and last 1 in each sorted sequence and replace with 2
    for sequence in sorted_sequences:
        # Find indices where the value is 1
        one_indices = (sequence == 1).nonzero(as_tuple=False).squeeze()
        if one_indices.numel() > 0:
            # Replace first and last 1 with 2
            sequence[one_indices[0]] = 2
            sequence[one_indices[-1]] = 2

    # print(sorted_sequences) 
    return sorted_sequences  

@torch.no_grad()
def main():
    parser = get_parser()

    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-decode-sentence")
    logging.info("Decoding started")

    device = torch.device("cpu")
    rank = 0 # hardcode 0 to use single GPU firstly
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    # add <SOS>, <EOS>, <PAD> token?
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

    token_ids, valid_ids, label_lens, label_masks = encode_sentences(params.text_file, sp, device)

    print(f"token_ids shape:{token_ids.shape}")
    print(f"-----------> token_ids:{token_ids}")
    print(f"-----------> valid_ids:{valid_ids}")
    print(f"-----------> label_lens:{label_lens}")

    active_case_logits, active_punct_logits, _ = model(token_ids, valid_ids=valid_ids, label_lens=label_lens)  
    # case_logits, punct_logits = model(token_ids, valid_ids=valid_ids, label_lens=label_lens, label_masks=label_masks)  
    # # print(f"case_logits shape:{case_logits.shape}, punct_logits shape:{punct_logits.shape}")

    # label_ids = None
    # label_lens, token_ids, label_ids, label_masks, valid_ids = sort_batch(label_lens, token_ids, label_ids, label_masks, valid_ids)
    # label_masks = label_masks[:, :case_logits.shape[1]]
    # active_ones = label_masks.reshape(-1) == 1

    # active_case_logits = case_logits.view(-1, params.out_size_case)[active_ones]
    # active_punct_logits = punct_logits.view(-1, params.out_size_punct)[active_ones]

    # # tensor0 = token_ids[0]
    # # decoded_text = sp.decode(tensor0.tolist())
    # # print(f"sp decode:{decoded_text}")
    # # print(f"case_logits[0]:{case_logits[0]}, case_logits[1]:{case_logits[1]}, case_logits[2]:{case_logits[2]}, case_logits[3]:{case_logits[3]}, case_logits[4]:{case_logits[4]}")
    case_pred = torch.argmax(F.log_softmax(active_case_logits, dim=1), dim=1)
    punct_pred = torch.argmax(F.log_softmax(active_punct_logits, dim=1), dim=1)
    # # print(f"after argmax:case_logits[0]:{case_logits[0]}, case_logits[1]:{case_logits[1]}, case_logits[2]:{case_logits[2]}, case_logits[3]:{case_logits[3]}, case_logits[4]:{case_logits[4]}")
    
    # # for i,x in enumerate(case_logits):
    # #     if not torch.equal(x, torch.tensor(1, device='cuda:3')):
    # #         print(f"case, index[{i}]:{x}")
    # # for i,x in enumerate(punct_logits):
    # #     if not torch.equal(x, torch.tensor(0, device='cuda:3')):
    # #         print(f"punct, index[{i}]:{x}")

    # handled_valid_ids = handle_bos_eos(valid_ids)
    # handled_valid_ids = handled_valid_ids[:, :case_logits.shape[1]]
    # flatten_valid_ids = handled_valid_ids.reshape(-1)
    # flatten_valid_ids = flatten_valid_ids[flatten_valid_ids != 0]
    # text_token_ones = flatten_valid_ids == 1
    # case_pred = case_pred[text_token_ones]
    # punct_pred = punct_pred[text_token_ones]

    logging.info(
            f"Result ==> "
            f"case pred:{case_pred}, "
            f"punct pred:{punct_pred}"
        )

    decode_sentences(params.text_file, case_pred.cpu(), punct_pred.cpu())


if __name__ == "__main__":
    main()