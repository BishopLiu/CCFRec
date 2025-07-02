import os
import argparse
import warnings
import torch
import random
import numpy as np
from data import load_split_data
from data import CCFSeqSplitDataset, Collator
from torch.utils.data import DataLoader
from model import CCFRec
from trainer import CCFTrainer
from utils import init_logger, init_seed, get_local_time, log, get_file_name, load_json, combine_index
from logging import getLogger
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument("--dataset", type=str, default="Musical_Instruments", help="dataset name")
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--tau', type=float, default=0.07, help='temperature')
    parser.add_argument('--cl_weight', type=float, default=0.1, help='contrastive loss weight')
    parser.add_argument('--mlm_weight', type=float, default=0.1, help='mlm weight')
    parser.add_argument('--neg_num', type=int, default=49)
    parser.add_argument('--text_types', nargs='+', type=str, default="meta")

    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_step', type=int, default=1, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="./dataset/", help="Input data path.")
    parser.add_argument('--map_path', type=str, default=".emb_map.json")
    parser.add_argument('--text_index_path', type=str, default=".code.pq.64_128.json")
    parser.add_argument('--text_emb_path', type=str, default=".t5.meta.emb.npy")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--n_codes_per_lel', type=int, default=256, help="number of codes per level")
    parser.add_argument('--code_level', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_layers_cross', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.5)
    parser.add_argument('--dropout_prob_cross', type=float, default=0.3)
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument('--metrics', type=str,
                        default="recall@5,ndcg@5,recall@10,ndcg@10")
    parser.add_argument('--valid_metric', type=str, default="ndcg@10")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--ckpt_dir", type=str,
                        default="./myckpt/",
                        help="output directory for model")
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":
    args = parse_arguments()
    args.run_local_time = get_local_time()
    args_dict = vars(args)

    args.save_file_name = get_file_name(args_dict) + \
                          f"_mlm{args.mlm_weight}_cl{args.cl_weight}_maskratio{args.mask_ratio}_drop{args.dropout_prob}_dpcross{args.dropout_prob_cross}"

    init_seed(args.seed, True)
    init_logger(args_dict, args.save_file_name+'.log')

    logger = getLogger()
    
    log(args_dict, logger)
    
    device = torch.device(args.device)
    
    data_path = args.data_path
    dataset = args.dataset
    dataset_path = os.path.join(data_path, dataset)

    item2id, n_items, train, val, test = load_split_data(args)
    index = load_json(os.path.join(dataset_path, dataset + args.text_index_path))

    train_dataset = CCFSeqSplitDataset(args, n_items, train, index, 'train')
    val_dataset = CCFSeqSplitDataset(args, n_items, val, index, 'val')
    test_dataset = CCFSeqSplitDataset(args, n_items, test, index, 'test')
    collator = Collator(args)

    train_data_loader = DataLoader(train_dataset, num_workers=args.num_workers, collate_fn=collator,
                                batch_size=args.batch_size, shuffle=True, pin_memory=False)

    val_data_loader = DataLoader(val_dataset, num_workers=args.num_workers, collate_fn=collator,
                                    batch_size=args.batch_size, shuffle=False, pin_memory=False)

    test_data_loader = DataLoader(test_dataset, num_workers=args.num_workers, collate_fn=collator,
                                    batch_size=args.batch_size, shuffle=False, pin_memory=False)

    text_embs = []
    for ttype in args.text_types:
        if ttype not in ['meta', 'title', 'brand', 'features', 'categories', 'description']:
            raise Exception(f"{ttype} not in [meta, brand, title, features, categories, description]")

        text_emb_file = f".t5.{ttype}.emb.npy"
        text_emb = np.load(os.path.join(args.data_path, args.dataset, args.dataset + text_emb_file))
        text_emb = PCA(n_components=args.embedding_size, whiten=True).fit_transform(text_emb)
        text_embs.append(text_emb)
    args.text_embedding_size = text_embs[0].shape[-1]


    model = CCFRec(args, train_dataset, index, device).to(device)

    for i in range(len(args.text_types)):
        model.item_text_embedding[i].weight.data[1:] = torch.tensor(text_embs[i], dtype=torch.float32, device=device)

    trainer = CCFTrainer(args, model, train_data_loader, val_data_loader, test_data_loader, device)

    log(model, logger)

    best_score, best_results = trainer.fit()
    test_results = trainer.test()

    log(f"Best Validation Score: {best_score}", logger)
    log(f"Test Results: {test_results}", logger)
    