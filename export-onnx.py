import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import onnx
import torch
import torch.nn as nn
# import torch.ao.quantization.quantize_dynamic as torch_quantize_dynamic
from onnxruntime.quantization import QuantType, quantize_dynamic
from train import get_model, get_params
from utils import (AttributeDict, setup_logger)
from onnxsim import simplify
from onnx import numpy_helper, helper
from onnxconverter_common import float16


##### usage:
## python3 export-onnx.py --exp_dir ../output --batch 22500

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The experiment dir contains .pt")
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

def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)

def export_model(
	model: nn.Module,
	filename: str,
    max_seq_length, int = 200,
	opset_version: int = 11,
) -> None:
    token_ids = torch.ones(1, max_seq_length, dtype=torch.int32)
    valid_ids = torch.ones(1, max_seq_length, dtype=torch.int32)
    label_lens = torch.tensor([200], dtype=torch.int32)

    model = torch.jit.trace(model, (token_ids, valid_ids, label_lens))

    torch.onnx.export(
                model,
				(token_ids, valid_ids, label_lens),
				filename,
                verbose=False,
				opset_version=opset_version,
                # do_constant_folding=False,
				input_names=["token_ids", "valid_ids", "label_lens"],
				output_names=["active_case_logits", "active_punct_logits", "mask"],
				dynamic_axes={
					"token_ids": {0: "N", 1: "T"},
					"valid_ids": {0: "N", 1: "T"},
					"label_lens": {0: "N"},
					"active_case_logits": {0: "Valid token ids num", 1: "case num"},
					"active_punct_logits": {0: "Valid token ids num", 1: "punct num"},
                    "mask": {0: "N", 1: "T'"},
				},
    )

    meta_data = {
        "NO_PUNCT": "0",
        "COMMA": "1",
        "PERIOD": "2",
        "QUESTION": "3",
        "LOWER": "0",
        "UPPER": "1",
        "CAP": "2",
        "MIX_CASE": "3",        
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    # print("setup logger")

    setup_logger(f"{params.exp_dir}/log-export")

    device = torch.device("cpu")
    # if torch.cuda.is_available():
    # 	device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    # print("after get model")

    model.to(device)

    if params.epoch > 0 :
        ptfile = f"{params.exp_dir}/epoch-{params.epoch-1}.pt"
    else:
        ptfile = f"{params.exp_dir}/checkpoint-{params.batch}.pt"
    logging.info(f"Loading checkpoint from {ptfile}")
    checkpoint = torch.load(ptfile, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    checkpoint.pop("model")

    model.to("cpu")
    model.eval()

    opset_version = 13

    logging.info("Exporting model")
    model_filename = params.exp_dir / f"model.onnx"
    export_model(
    	model, 
    	model_filename,
        max_seq_length = params.max_seq_length,
    	opset_version = opset_version,
    )
    logging.info(f"Exported model to {model_filename}")


    onnx_model = onnx.load(model_filename)
    model_sim, check = simplify(onnx_model)
    model_sim_filename = params.exp_dir / f"model_sim.onnx"
    onnx.save(model_sim, model_sim_filename)
    logging.info(f"Exported simplified model to {model_sim_filename}")

    model_filename_int8 = params.exp_dir / f"model.int8.onnx"
    quantize_dynamic(
        model_input=model_sim_filename,
        model_output=model_filename_int8,
        weight_type=QuantType.QUInt8,
    )
    logging.info(f"Exported quantized model to {model_filename_int8}")

    # model_fp16_filename = params.exp_dir / f"model_fp16.onnx"
    # model = onnx.load(model_sim_filename)
    # model_fp16 = float16.convert_float_to_float16(model)
    # onnx.save(model_fp16, model_fp16_filename)



if __name__ == "__main__":
    main()