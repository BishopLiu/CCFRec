import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
import math
import torch.nn as nn
from colorama import init
from utils import ensure_dir, set_color, get_local_time
import os
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from metrics import metrics_to_function
from logging import getLogger
from utils import log
from typing import Dict
from collections import defaultdict
import torch.nn.functional as F
init(autoreset=True)


class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x, y):

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        B = x.shape[0]

        logits = torch.matmul(x, y.transpose(0, 1)) / self.tau
        labels = torch.arange(B, device=x.device, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)
        return loss


class BaseTrainer(object):
    def __init__(self, args, model, train_data, valid_data=None, test_data=None, device=None):
        self.args = args
        self.model = model
        self.logger = getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.eval_step = min(args.eval_step, self.epochs)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.all_metrics = args.metrics.split(",")
        self.valid_metric = args.valid_metric
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.max_steps = self.get_train_steps()
        self.warmup_steps = args.warmup_steps
        self.optimizer = self._build_optimizer()
        if self.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=self.warmup_steps,
                                                                num_training_steps=self.max_steps)
        else:
            self.lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                                  num_warmup_steps=self.warmup_steps)

        self.device = device

        self.ckpt_dir = args.ckpt_dir
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.args.dataset, args.save_file_name)
        ensure_dir(self.ckpt_dir)
        self.best_score = 0
        self.best_ckpt = "best_model.pth"

    def _build_optimizer(self):
        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def get_train_steps(self):
        len_dataloader = len(self.train_data)
        num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)

        return max_steps

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, epoch_idx, verbose=True):

        self.model.train()

        total_num = 0
        total_loss = 0
        iter_data = tqdm(
                    self.train_data,
                    total=len(self.train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    disable=not verbose,
                    )

        for batch_idx, data in enumerate(iter_data):
            item_inters, inter_lens, labels \
                = data["item_inters"].to(self.device), data["inter_lens"].to(self.device), data["targets"].to(self.device)

            total_num += 1

            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(item_inters, inter_lens, labels)
    
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item()
            iter_data.set_postfix(loss=loss.item())

        return total_loss/total_num

    def evaluate(self, scores, labels):

        metrics = {m:0 for m in self.all_metrics}

        _, topk_idx = torch.topk(
            scores, self.max_topk, dim=-1
        )  # B x k
        topk_idx = topk_idx.detach().cpu()
        labels = labels.detach().cpu()

        one_hot_labels = torch.zeros_like(scores).detach().cpu()
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        top_k_labels = torch.gather(one_hot_labels, dim=1, index=topk_idx).numpy()
        pos_nums = one_hot_labels.sum(dim=1).numpy()

        topk_metrics = {}
        for m_name in self.all_metric_name:
            value = metrics_to_function[m_name](top_k_labels, pos_nums)
            topk_metrics[m_name] = value.sum(axis=0)

        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            m_name = m_name.lower()
            top_k = int(top_k)
            value = topk_metrics[m_name]
            metrics[m] = value[top_k - 1]

        return metrics

    def _save_checkpoint(self, epoch, ckpt_file=None, verbose=True):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, self.best_ckpt)
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_score": self.best_score,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        if verbose:
            self.log(f"[Epoch {epoch}] Saving current: {ckpt_path}")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        train_loss_output = ("[Epoch %d] training [time: %.2fs, "
        ) % (epoch_idx, e_time - s_time)

        if isinstance(loss, float):
            train_loss_output += "train loss: %.4f" % loss
        elif isinstance(loss, Dict) or isinstance(loss, defaultdict):
            for k, v in loss.items():
                train_loss_output += f"{k}: {v:.4f}, "
        return train_loss_output + "]"

    def fit(self, verbose=True):
        cur_eval_step = 0
        stop = False
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(epoch_idx, verbose=verbose)
            training_end_time = time()

            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.log(train_loss_output)

            if (epoch_idx + 1) % self.eval_step == 0:
                metrics = self._test_epoch(test_data=self.valid_data, verbose=verbose)
                total_metrics = metrics

                if total_metrics[self.valid_metric] > self.best_score:
                    self.best_score = total_metrics[self.valid_metric]
                    self.best_result = total_metrics
                    cur_eval_step = 0
   
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                else:
                    cur_eval_step += 1

                if cur_eval_step >= self.early_stop:
                    stop = True
                if verbose:
                    self.log(f"[Epoch {epoch_idx}] Val Result: {total_metrics}")

            if stop:
                break

        return self.best_score, self.best_result

    @torch.no_grad()
    def test(self,verbose=True):
        test_results=None
        if self.test_data is not None:
            metrics = self._test_epoch(load_best_model=True, verbose=verbose)
            test_results = metrics
        return test_results

    @torch.no_grad()
    def _test_epoch(self, test_data=None, load_best_model=False, model_file=None, verbose=True):
        
        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            checkpoint_file = model_file or os.path.join(self.ckpt_dir, self.best_ckpt)
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            message_output = "Loading model parameters from {}".format(
                checkpoint_file
            )
            if verbose:
                self.log(message_output)

        self.model.eval()

        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=not verbose,
        )

        total = 0
        metrics = {m: 0 for m in self.all_metrics}
        for batch_idx, data in enumerate(iter_data):
            item_inters, inter_lens, labels \
                = data["item_inters"].to(self.device), \
                  data["inter_lens"].to(self.device), data["targets"].to(self.device)

            total += len(labels)
            scores = self.model.full_sort_predict(item_inters, inter_lens)

            _metrics = self.evaluate(scores, labels)
            for m, v in _metrics.items():
                metrics[m] += v

        for m in metrics:
            metrics[m] = metrics[m] / total

        return metrics

    def log(self, message, level='info'):
        return log(message, self.logger, level=level)


class CCFTrainer(BaseTrainer):
    def __init__(self, args, model, train_data, valid_data=None, test_data=None, device=None):
        super(CCFTrainer, self).__init__(args, model, train_data, valid_data, test_data, device)

    def _train_epoch(self, epoch_idx, verbose=True):

        self.model.train()
        
        total_num = 0
        total_loss = defaultdict(float)
        iter_data = tqdm(
                    self.train_data,
                    total=len(self.train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    disable=not verbose,
                    )

        for batch_idx, data in enumerate(iter_data):

            item_inters, inter_lens, labels, code_inters, mask_labels = \
                    data["item_inters"].to(self.device), data["inter_lens"].to(self.device), data["targets"].to(self.device),\
                    data['code_inters'].to(self.device), data["mask_targets"].to(self.device)

            total_num += 1

            self.optimizer.zero_grad()

            loss = self.model.calculate_loss(item_inters, inter_lens, labels,
                                             code_inters, mask_labels)
    
            self._check_nan(loss['loss'])
            loss['loss'].backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            iter_data.set_postfix(loss=loss['loss'].item())
            for k in loss.keys():
                total_loss[k] += loss[k].item()

        for k in total_loss.keys():
            total_loss[k] /= total_num
        return total_loss

    @torch.no_grad()
    def _test_epoch(self, test_data=None, load_best_model=False, model_file=None, verbose=True):

        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            checkpoint_file = model_file or os.path.join(self.ckpt_dir, self.best_ckpt)
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            message_output = "Loading model parameters from {}".format(
                checkpoint_file
            )
            if verbose:
                self.log(message_output)

        self.model.eval()
        self.model.get_item_embedding()

        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=not verbose,
        )

        total = 0
        metrics = {m: 0 for m in self.all_metrics}
        for batch_idx, data in enumerate(iter_data):
            item_inters, inter_lens, code_inters, labels \
                = data["item_inters"].to(self.device), data["inter_lens"].to(self.device), \
                  data['code_inters'].to(self.device), data["targets"].to(self.device)

            total += len(labels)
            scores = self.model.full_sort_predict(item_inters, inter_lens, code_inters)

            _metrics = self.evaluate(scores, labels)
            for m, v in _metrics.items():
                metrics[m] += v

        for m in metrics:
            metrics[m] = metrics[m] / total

        return metrics

