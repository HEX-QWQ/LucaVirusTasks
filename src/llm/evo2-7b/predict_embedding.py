#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey,Majiahao
@email: 
@tel: 
@datetime: 2025/9/10
@project: LucaVirusTasks
@file: inference_embedding
@desc: embedding inference for Evo2-1b
'''
import os
import sys
import torch
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader, csv_reader, tsv_reader
    from utils import clean_seq_luca, calc_emb_filename_by_seq_id
except ImportError:
    from src.file_operator import fasta_reader, csv_reader, tsv_reader
    from src.utils import clean_seq_luca, calc_emb_filename_by_seq_id

from evo2 import Evo2

model_id = 'evo2_7b'
layer_name = "blocks.28.mlp.l3"

evo_global_model, evo_global_alphabet, evo_global_version = None, None, None


def predict_embedding(
        sample,
        trunc_type,
        embedding_type,
        repr_layers=[-1],
        truncation_seq_length=4094,
        device=None,
        version="evo",
        matrix_add_special_token=False,
        fp16=False
):
    '''
    use sequence to predict protein embedding matrix or vector(bos)
    :param sample: [protein_id, protein_sequence]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094,2046,1982,1790,1534,1278,1150,1022]
    :param device:
    :param version:
    :param matrix_add_special_token:
    :param fp16:
    :return: embedding, processed_seq_len
    '''
    
    global evo_global_model, evo_global_alphabet, evo_global_version
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    if len(sample) > 2:
        seq_id, seq = sample[0], sample[2]
    else:
        seq_id, seq = sample[0], sample[1]
    processed_seq = clean_seq_luca(seq_id, seq)
    if len(processed_seq) > truncation_seq_length:
        if trunc_type == "left":
            processed_seq = processed_seq[-truncation_seq_length:]
        else:
            processed_seq = processed_seq[:truncation_seq_length]
    if evo_global_model is None or evo_global_alphabet is None or evo_global_version is None or evo_global_version != version:
        if version == "evo":
            evo_global_model = Evo2(model_id)
        else:
            raise Exception("not support this version=%s" % version)
        evo_global_version = version
    '''
    if torch.cuda.is_available() and device is not None:
        evo_global_model = evo_global_model.to(device)
    elif torch.cuda.is_available():
        evo_global_model = evo_global_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("llm use cpu")
    '''
    # evo_global_model = evo_global_model.to(device)
    # evo_global_model.eval()

    inputs = torch.tensor(
        evo_global_model.tokenizer.tokenize(processed_seq),  # 这里的 tokenizer 来自 model.tokenizer
        dtype=torch.int
    ).unsqueeze(0)
    embeddings = {}
    with torch.no_grad():
        # if torch.cuda.is_available():
        inputs = inputs.to(device=device, non_blocking=True)
        try:
            _,out = evo_global_model(inputs,return_embeddings=True,layer_names=[layer_name])
            truncate_len = min(truncation_seq_length, inputs.shape[1] - 2)
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_token:
                    embedding = out[layer_name][0, 0: truncate_len + 2].cpu().to(torch.float16).clone().numpy()
                else:
                    embedding = out[layer_name][0, 1: truncate_len + 1].cpu().to(torch.float16).clone().numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out.to(device="cpu")[0, 0].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                # to do
                embeddings["contacts"] = None
            if len(embeddings) > 1:
                return embeddings, processed_seq
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], processed_seq
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
                print("Please reduce the 'truncation_seq_length'")
            raise Exception(e)
    return None, None


def get_args():
    parser = argparse.ArgumentParser(description='evo Embedding')
    # for one seq
    parser.add_argument("--seq_id", type=str, default=None,
                        help="the seq id")
    parser.add_argument("--seq", type=str, default=None,
                        help="when to input a seq")
    parser.add_argument("--seq_type", type=str, default="gene",
                        choices=["gene", "multi_gene"],
                        help="the input seq type")

    # for many
    parser.add_argument("--input_file", type=str, default=None,
                        help="the input filepath(.fasta or .csv or .tsv)")

    # for input csv
    parser.add_argument("--id_idx", type=int, default=None,
                        help="id col idx(0 start)")
    parser.add_argument("--seq_idx", type=int, default=None,
                        help="seq col idx(0 start)")

    # for saved path
    parser.add_argument("--save_path", type=str, default=None,
                        help="embedding file save dir path")

    parser.add_argument("--embedding_type", type=str, default="matrix",
                        choices=["matrix", "vector"],
                        help="the llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right",
                        choices=["left", "right"],
                        help="llm trunc type.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094,
                        help="the llm truncation seq length(not contain [CLS] and [SEP].")
    parser.add_argument("--matrix_add_special_token", action="store_true",
                        help="whether to add special token embedding vector in seq representation matrix")
    parser.add_argument("--embedding_complete",
                        action="store_true",
                        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")

    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="the gpu id to use.")

    input_args = parser.parse_args()
    return input_args


def main(model_args):
    print(model_args)
    if model_args.gpu_id >= 0:
        gpu_id = model_args.gpu_id
    else:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        print("gpu_id: ", gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    assert (model_args.input_file is not None and os.path.exists(model_args.input_file)) or model_args.seq is not None
    print("input seq type: %s" % model_args.seq_type)
    print("args device: %s" % model_args.device)
    embedding_type = model_args.embedding_type
    seq_type = model_args.seq_type
    emb_save_path = model_args.save_path
    print("emb save dir: %s" % os.path.abspath(emb_save_path))
    if seq_type not in ["gene", "multi_gene"]:
        print("Error! arg: --seq_type=%s is not 'gene' or 'multi_gene'" % seq_type)
        sys.exit(-1)
    input_seq_type = seq_type
    if seq_type == "multi_gene":
        seq_type = "gene"
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)
    if model_args.input_file:
        done = 0
        file_reader = fasta_reader
        if model_args.input_file.endswith(".csv"):
            file_reader = csv_reader
        elif model_args.input_file.endswith(".tsv"):
            file_reader = tsv_reader

        for row in file_reader(model_args.input_file):
            if model_args.id_idx is None or model_args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[model_args.id_idx].strip(), row[model_args.seq_idx].upper()
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            if not os.path.exists(embedding_filepath):
                emb_list = []
                # multi seqs
                if "," in seq:
                    multi_seqs = [v.strip().upper() for v in seq.split(",")]
                else:
                    multi_seqs = [seq.strip().upper()]
                assert len(multi_seqs) == 1 or input_seq_type in ["multi_gene"]
                for seq in multi_seqs:
                    ori_seq_len = len(seq)
                    truncation_seq_length = model_args.truncation_seq_length
                    if model_args.embedding_complete:
                        truncation_seq_length = ori_seq_len
                    emb, processed_seq_len = predict_embedding(
                        [seq_id, seq_type, seq],
                        model_args.trunc_type,
                        embedding_type,
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        device=model_args.device,
                        version="evo",
                        matrix_add_special_token=model_args.matrix_add_special_token
                    )
                    while emb is None:
                        print("%s embedding error, max_len from %d truncate to %d" % (
                            seq_id,
                            truncation_seq_length,
                            int(truncation_seq_length * 0.95)
                        ))
                        truncation_seq_length = int(truncation_seq_length * 0.95)
                        emb, processed_seq_len = predict_embedding(
                            [seq_id, seq_type, seq],
                            model_args.trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device,
                            version="evo",
                            matrix_add_special_token=model_args.matrix_add_special_token
                        )
                    # print("seq_len: %d" % len(seq))
                    # print("emb shape:", embedding_info.shape)
                    # torch.save(emb, embedding_filepath)
                    emb_list.append(emb)
                if input_seq_type in ["multi_gene"]:
                    torch.save(emb_list, embedding_filepath)
                else:
                    torch.save(emb_list[0], embedding_filepath)
            else:
                print("%s exists." % embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)
        print("embedding over, done: %d" % done)
    elif model_args.seq:
        print("input seq length: %d" % len(model_args.seq))
        emb, processed_seq_len = predict_embedding(
            ["input", model_args.seq],
            model_args.trunc_type,
            model_args.embedding_type,
            repr_layers=[-1],
            truncation_seq_length=model_args.truncation_seq_length,
            device=model_args.device,
            version="evo",
            matrix_add_special_token=model_args.matrix_add_special_token
        )
        print("done seq length: %d" % processed_seq_len)
        print(emb)
        if emb is not None:
            print(emb.shape)


if __name__ == "__main__":
    run_args = get_args()
    main(run_args)