import os
import numpy as np
import torch
from utils import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_split_data(args):
    def transform_token2id_seq(item_seqs, item2id):
        id_seqs = []
        for one_piece in item_seqs:
            item_seq = one_piece["inter_history"]
            item_id_seq = [item2id[item] for item in item_seq]

            target_id = item2id[one_piece["target_id"]]
            id_seqs.append(item_id_seq + [target_id])
        return id_seqs

    data_path = args.data_path
    dataset = args.dataset
    dataset_path = os.path.join(data_path, dataset)

    all_items = load_json(os.path.join(dataset_path, dataset + ".item.json"))

    item2id = load_json(os.path.join(dataset_path, dataset + args.map_path))

    train_inter = load_jsonl(os.path.join(dataset_path, dataset + ".train.jsonl"))
    valid_inter = load_jsonl(os.path.join(dataset_path, dataset + ".valid.jsonl"))
    test_inter = load_jsonl(os.path.join(dataset_path, dataset + ".test.jsonl"))

    train_seq = transform_token2id_seq(train_inter, item2id)
    valid_seq = transform_token2id_seq(valid_inter, item2id)
    test_seq = transform_token2id_seq(test_inter, item2id)

    n_items = len(all_items)

    return item2id, n_items, train_seq, valid_seq, test_seq


class CCFSeqSplitDataset(Dataset):
    def __init__(self, args, n_items, inter_seq, index, mode="train"):
        self.n_items = n_items  # no padding
        self.args = args
        self.max_his_len = args.max_his_len
        self.n_digits = args.code_level
        self.n_codes = args.n_codes_per_lel * args.code_level
        self.index = index
        self.mask_token_id = 0
        self.mode = mode
        self.mask_ratio = args.mask_ratio
        self.inter_data = self.__map_inter__(inter_seq)

    def __map_inter__(self, inter_seq):
        inter_data = []
        for seq in inter_seq:
            item_seq = seq[:-1][-self.max_his_len:]
            code_seq = []
            for item in item_seq:
                id = []
                for i in range(self.n_digits):
                    id.append(self.index[item][i]+i*self.args.n_codes_per_lel+1)
                code_seq.extend(id)

            inter_data.append({"item_inter": item_seq, "code_inter": code_seq, "target": seq[-1]})

        return inter_data

    def __getitem__(self, idx):
        data = self.inter_data[idx]
        item_inter = data['item_inter']
        code_inter = np.array(data['code_inter'])

        target = data["target"]

        mask_target = np.ones_like(code_inter) * -100

        if self.mode == "train":
            L = len(item_inter)
            code_inter, mask_target = self.__mask__(code_inter.reshape(L, -1))

        return item_inter, target, code_inter, mask_target

    def __mask__(self, code_inter):
        BL, C = code_inter.shape[0], code_inter.shape[1]
        mask_target = np.ones_like(code_inter) * -100

        mask = np.random.rand(BL, C) < self.mask_ratio
        mask_idx, mask_idy = np.where(mask)
        mask_target[mask] = code_inter[mask]
        for x, y in zip(mask_idx, mask_idy):
            rand = np.random.rand()
            if rand < 0.8:
                code_inter[x, y] = self.mask_token_id
            elif rand < 0.9:
                code_inter[x, y] = np.random.randint(1, self.n_codes+1)
            else:
                pass

        code_inter = code_inter.reshape(-1)
        mask_target = mask_target.reshape(-1)

        return code_inter, mask_target

    def __len__(self):
        return len(self.inter_data)


class Collator(object):
    def __init__(self, args):
        self.n_digits = args.code_level
        self.max_his_len = args.max_his_len

    def __call__(self, batch):
        item_inters, targets, code_inters, mask_targets = zip(*batch)

        inter_lens = get_seqs_len(item_inters)

        item_inters = [torch.tensor(inter) for inter in item_inters]
        item_inters = pad_sequence(item_inters).transpose(0, 1)

        code_inters = [torch.tensor(inter) for inter in code_inters]
        code_inters = pad_sequence(code_inters).transpose(0, 1)
        code_inters = code_inters.reshape(-1, self.n_digits)

        targets = torch.tensor(targets)

        mask_targets = [torch.tensor(mask_target) for mask_target in mask_targets]
        mask_targets = pad_sequence(mask_targets, padding_value=-100).transpose(0, 1)

        mask_targets = mask_targets.reshape(-1)


        return dict(item_inters=item_inters.long(),
                    code_inters_a=code_inters.long(),
                    inter_lens=inter_lens.long(),
                    targets=targets.long(),
                    mask_targets_a=mask_targets.long())

