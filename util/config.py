import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = None
    dim: int = 768
    seq_len: int = 128
    n_segments: int = 2
    drop_out: float = 0.1
    n_layers: int = 12
    n_heads: int = 12
    n_hidden: int = 3072
    mode: str = 'train'
    save_model_prefix: str = ''
    seed: int = 100
    save_model_prefix: str = 'save/model/prefix'
    prune_emb: bool = False
    emb_factor: int = 1
    freeze_embeddings: bool = True
    masking: bool = False
    prune_attn: bool = True
    prune_ff: bool = True

    @classmethod
    def parse_json(cls, path):
        return cls(**json.load(open(path, "r")))

@dataclass(frozen=True)
class TrainingConfig:
    task: str = 'mrpc'
    epoch: int = 100
    lr: float = 0.01
    vocab: str = ''
    batch_size: int = 64
    data_parallel: bool = True
    kl_fac: float = 0.001
    ib_lr: float = 0.001
    warm_up: float = 0.1
    pretrain_model_path: str = 'path/to/pretrain/model'
    save_steps: int = 1000
    pruned_model_path: str = 'path/to/pruned/model'

    @classmethod
    def parse_json(cls, path):
        return cls(**json.load(open(path, "r")))

@dataclass(frozen=True)
class EvaluateConfig:
    task: str = 'mrpc'
    kl_fac: float = 0.001
    lr: float = 0.001
    ib_lr: float = 0.001
    eval_steps: int = 1000
    epoch: int = 10
    ib_threshold: int = 0
    data_parallel: bool = True

    @classmethod
    def parse_json(cls, path):
        return cls(**json.load(open(path, "r")))


