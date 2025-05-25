import faiss
import json
import numpy as np
import argparse
import yaml
import os
from collections import defaultdict
from sklearn.decomposition import PCA


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="instrument.yaml")
    parser.add_argument('--type', type=str, default='pq', help='pq or rq')

    args, _ = parser.parse_known_args()
    return args


def main(config):
    n_bits = int(np.log2(config['code_num']))
    dataset_name = config['dataset']
    data_path = os.path.join(config['data_path'], dataset_name)
    text_types = config['text_types']
    sem_ids_path = config['semantic_ids_path'] + \
    f"{dataset_name}.code.{args.type}.{config['n_codebooks']}_{config['code_num']}.pca{config['pca_size']}." + \
                   "_".join(text_types) + ".json"
    sem_emb_path_format = "{}.t5.{}.emb.npy"

    sent_embs = []
    for text in text_types:
        sem_emb_path = os.path.join(data_path, sem_emb_path_format.format(dataset_name, text))
        sent_emb = np.load(sem_emb_path)

        if config['pca_size']>0:
            pca = PCA(n_components=config['pca_size'], whiten=True)
            sent_emb = pca.fit_transform(sent_emb)
            sent_emb = np.ascontiguousarray(sent_emb)
            print("PCA done")
        sent_embs.append(sent_emb)
        
    faiss.omp_set_num_threads(config['faiss_omp_num_threads'])
    type_num = len(text_types)
    
    if args.type == 'pq':
        sent_embs = np.concatenate(sent_embs, axis=1)
        
        print(sent_embs.shape)

        index = faiss.IndexPQ(sent_embs.shape[-1], config['n_codebooks'], n_bits, faiss.METRIC_INNER_PRODUCT)

        print(f'[TOKENIZER] Training index...')
        index.train(sent_embs)
        index.add(sent_embs)
        faiss_sem_ids = []
        uint8_code = index.pq.compute_codes(sent_embs)
        n_bytes = uint8_code.shape[1]
        print(f'[TOKENIZER] Generating semantic IDs...')
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            code = []
            for i in range(config['n_codebooks']):
                code.append(bs.read(n_bits))
            faiss_sem_ids.append(code)
        faiss_sem_ids = np.array(faiss_sem_ids)
        item2sem_ids = [[]] + faiss_sem_ids.tolist()
    elif args.type == 'rq':
        sem_ids_list = []
        for sent_emb in sent_embs:
            index = faiss.IndexResidualQuantizer(
                    sent_emb.shape[-1],
                    config['n_codebooks'],
                    n_bits,
                    faiss.METRIC_INNER_PRODUCT
                    )
            print(f'[TOKENIZER] Training index...')
            index.train(sent_emb)
            index.add(sent_emb)
            faiss_sem_ids = []
            uint8_code = index.rq.compute_codes(sent_emb)
            n_bytes = uint8_code.shape[1]
            print(f'[TOKENIZER] Generating semantic IDs...')
            for u8_code in uint8_code:
                bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
                code = []
                for i in range(config['n_codebooks']):
                    code.append(bs.read(n_bits))
                faiss_sem_ids.append(code)
            faiss_sem_ids = np.array(faiss_sem_ids)
            sem_ids_list.append(faiss_sem_ids.tolist())
        item2sem_ids = [[]]
        item_num = len(sem_ids_list[0])
        for i in range(item_num):
            sem_ids = []
            for j in range(type_num):
                sem_ids.extend(sem_ids_list[j][i])
            item2sem_ids.append(sem_ids)
    else:
        raise ValueError(f'Invalid type: {args.type}')
        
    print(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
    with open(sem_ids_path, 'w') as f:
        json.dump(item2sem_ids, f)


if __name__ == '__main__':
    args = parse_arguments()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)
