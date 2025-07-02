import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--data_path', type=str, default="./dataset")
    parser.add_argument('--text_types', type=str, nargs="+", default="title")

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset
    dataset_path = os.path.join(args.data_path, dataset)
    
    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)
    
    
    meta_text_dict = json.load(open(os.path.join(dataset_path, f'{dataset}.meta.json'), 'r'))
    attr_list = args.meta_types

    for attr in attr_list:
        if attr not in meta_text_dict["1"].keys():
            raise ValueError(f"Attribute {attr} not found in meta data.")
        sorted_text = [v[attr] for k, v in sorted(meta_text_dict.items(), key=lambda x: int(x[0]))] 
        
        
        text_embedding_model = SentenceTransformer('sentence-t5-base').to(device)
        embs = text_embedding_model.encode(sorted_text, convert_to_numpy=True, batch_size=256, show_progress_bar=True)

        print(embs.shape)
        np.save(dataset_path + f'{dataset}.t5.{attr}.emb.npy', embs)
        print(f"Saved {dataset}.t5.{attr}.emb.npy")

