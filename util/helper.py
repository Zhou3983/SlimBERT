import random
import numpy as np
import torch
import torch.nn as nn
import checkpoint
import os
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

LOSS = nn.CrossEntropyLoss()


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_loss(model, batch, kl_fac, model_config):  # make sure loss is a scalar tensor
    input_ids, segment_ids, input_mask, label_id = batch
    logits = model(input_ids, segment_ids, input_mask, model_config.prune_emb)
    model_loss = LOSS(logits, label_id)
    if model_config.masking:
        return model_loss, model_loss, 0
    else:
        ib_loss = model.transformer.compute_slim_loss(model_config)
        loss = model_loss + ib_loss * kl_fac
        return loss, model_loss, ib_loss


def load(model, model_file=None, pretrain_file=None):
    """ load saved model or pretrained transformer (a part of model) """
    if model_file is not None:
        print('Loading the model from', model_file)
        model.load_state_dict(torch.load(model_file), strict=False)

    elif pretrain_file is not None:  # use pretrained transformer
        print('Loading the pretrained model from', pretrain_file)
        if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
            checkpoint.load_model(model.transformer, pretrain_file)
        elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
            model.transformer.load_state_dict(
                {key[12:]: value
                 for key, value in torch.load(pretrain_file).items()
                 if key.startswith('transformer')}, strict=False
            )  # load only transformer parts
    return model


def load_neuron_importance(model, pruned_model_file):
    print('Loading the neuron importance from', pruned_model_file)
    model.transformer.load_state_dict({key[12:]: value
                 for key, value in torch.load(pruned_model_file).items()
                 if 'ib' in key and key.endswith('mu')}, strict=False)
    return model


def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def save_model(model, i, train_config, model_config, param_percent=None):
    """ save current model """
    emb_flag = "prune" if model_config.prune_emb else "noprune"
    if param_percent is not None:
        torch.save(model.state_dict(),  # save model object before nn.DataParallel
                   os.path.join(model_config.save_model_prefix,
                                f'{train_config.task}_lr{train_config.lr}_steps_{i}_{emb_flag}emb_{param_percent}.pt'))
    else:
        torch.save(model.state_dict(),  # save model object before nn.DataParallel
                   os.path.join(model_config.save_model_prefix,
                                f'{train_config.task}_KL{train_config.kl_fac}_lr{train_config.lr}'
                                                        f'_iblr{train_config.ib_lr}_steps_{i}_{emb_flag}emb.pt'))


def get_eval_path(eval_train_config, model_config, steps, param_percent=None):
    emb_flag = "prune" if model_config.prune_emb else "noprune"
    if param_percent is not None:
        return f"{model_config.save_model_prefix}/{eval_train_config.task}_lr{eval_train_config.lr}_steps_{steps}_{emb_flag}emb_{param_percent}.pt"
    else:
        return f"{model_config.save_model_prefix}/{eval_train_config.task}_KL{eval_train_config.kl_fac}_lr{eval_train_config.lr}" \
               f"_iblr{eval_train_config.ib_lr}_steps_{steps}_{emb_flag}emb.pt"


def get_partial_acc(model, batch):  # make sure loss is a scalar tensor
    input_ids, segment_ids, input_mask, label = batch
    logits = model(input_ids, segment_ids, input_mask)
    _, label_pred = logits.max(1)
    result = (label_pred == label).float()
    accuracy = result.mean()
    return accuracy, result


def get_batch_output(model, batch, prune_emb):
    input_ids, segment_ids, input_mask, label = batch
    logits = model(input_ids, segment_ids, input_mask, prune_emb)
    _, pred = logits.max(1)
    return label.tolist(), pred.tolist()

def compute_acc(data_iter, model, prune_emb, task):
    pred_ls = []
    true_ls = []
    device = get_device()
    for i, batch in enumerate(data_iter):
        batch = [data.to(device) for data in batch]
        with torch.no_grad():
            label, pred = get_batch_output(model, batch, prune_emb)  # mean() for Data Parallelism
        pred_ls += pred
        true_ls += label
    if task != "cola":
        acc = accuracy_score(true_ls, pred_ls)
        print('acc: ', acc)
        try:
            f1 = f1_score(true_ls, pred_ls)
            print(f'f1: {f1}')
        except:
            print('no f1')
    else:
        acc = accuracy_score(true_ls, pred_ls)
        print('acc: ', acc)
        mcc = matthews_corrcoef(true_ls, pred_ls)
        print('mcc: ', mcc)


def report_percentage_by_bert_layer_neurons(prune_stat):
    prune_percentage = []
    for i in range(len(prune_stat)):
        pruned_neurons = prune_stat[i]
        if i != 3:
            prune_percentage.append(pruned_neurons / (768 * 12))
        else:
            prune_percentage.append(pruned_neurons / (768 * 12 * 4))
    print(prune_percentage)


def report_percentage_by_pruned_neurons(prune_stat):
    sum_prune_stat = sum(prune_stat)
    prune_percentage = [item/sum_prune_stat for item in prune_stat]
    print(prune_percentage)


def report_compression_rate(modelConfig, model, prune_threshold):
    if modelConfig.prune_emb:
        prune_stat, remaining_params, remaining_params_include_emb = model.transformer.get_hard_masks(
            prune_emb=modelConfig.prune_emb, threshold=prune_threshold)

        print(f'prune_stat: {prune_stat}')
        report_percentage_by_bert_layer_neurons(prune_stat)
        report_percentage_by_pruned_neurons(prune_stat)
        print('compression rate in params W emb: ', remaining_params_include_emb / 108853248 * 100)
        print('compression rate in params WO emb: ', remaining_params / 85017600 * 100)
        compression_rate = sum(prune_stat) / (modelConfig.dim * modelConfig.n_layers * 7 + modelConfig.dim)
    else:
        prune_stat, remaining_params = model.transformer.get_hard_masks(prune_emb=modelConfig.prune_emb,
                                                                        threshold=prune_threshold)
        print(f'prune_stat: {prune_stat}')
        report_percentage_by_bert_layer_neurons(prune_stat)
        report_percentage_by_pruned_neurons(prune_stat)
        print('compression rate in params W emb: ', (remaining_params + 23835648) / 108853248 * 100)
        print('compression rate in params WO emb: ', remaining_params / 85017600 * 100)
        compression_rate = sum(prune_stat) / (modelConfig.dim * modelConfig.n_layers * 7)
    print('compression rate in neurons: ', (1 - compression_rate) * 100)