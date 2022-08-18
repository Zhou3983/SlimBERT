# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Load a checkpoint file of pretrained transformer to a model in pytorch """

import numpy as np
import tensorflow as tf
import torch


# import ipdb
# from models import *

def load_param(checkpoint_file, conversion_table):
    """
    Load parameters in pytorch model from checkpoint file according to conversion_table
    checkpoint_file : pretrained checkpoint model file in tensorflow
    conversion_table : { pytorch tensor in a model : checkpoint variable name }
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

        # for weight(kernel), we should do transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % \
            (tuple(pyt_param.size()), tf_param.shape, tf_param_name)

        # assign pytorch tensor from tensorflow param
        pyt_param.data = torch.from_numpy(tf_param)


def load_model(model, checkpoint_file):
    """ Load the pytorch model from checkpoint file """

    # Embedding layer
    e, p = model.embedding, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.tok_emb.weight: p + "word_embeddings",
        e.pos_emb.weight: p + "position_embeddings",
        e.seg_emb.weight: p + "token_type_embeddings",
        e.LN.gamma: p + "LayerNorm/gamma",
        e.LN.beta: p + "LayerNorm/beta"
    })

    # Transformer blocks
    for i in range(len(model.bottledBERT)):
        b, p = model.bottledBERT[i], f"bert/encoder/layer_{i}/"
        load_param(checkpoint_file, {
            b.attention.q.weight: p + "attention/self/query/kernel",
            b.attention.q.bias: p + "attention/self/query/bias",
            b.attention.k.weight: p + "attention/self/key/kernel",
            b.attention.k.bias: p + "attention/self/key/bias",
            b.attention.v.weight: p + "attention/self/value/kernel",
            b.attention.v.bias: p + "attention/self/value/bias",
            b.dense.weight: p + "attention/output/dense/kernel",
            b.dense.bias: p + "attention/output/dense/bias",
            b.FF.fc1.weight: p + "intermediate/dense/kernel",
            b.FF.fc1.bias: p + "intermediate/dense/bias",
            b.FF.fc2.weight: p + "output/dense/kernel",
            b.FF.fc2.bias: p + "output/dense/bias",
            b.LN1.gamma: p + "attention/output/LayerNorm/gamma",
            b.LN1.beta: p + "attention/output/LayerNorm/beta",
            b.LN2.gamma: p + "output/LayerNorm/gamma",
            b.LN2.beta: p + "output/LayerNorm/beta",
        })