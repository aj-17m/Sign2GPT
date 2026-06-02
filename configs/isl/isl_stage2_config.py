"""
Stage 2 (translation training) config for the ISL training pipeline.

Adapted from PHOENIX stage 2 config with:
  - Data paths point to /workspace/data/isl/ + /workspace/lmdb/isl/
  - cfg.stage1_name -> the ISL stage 1 config (not PHOENIX's)
  - 60 epochs (vs PHOENIX validation's 10; ISL needs more passes)
  - LLM is unchanged (XGLM-1.7B supports English natively)
"""

from ml_collections import config_dict
from pathlib import Path
import os
import numpy as np
import torch
from configs.base.base_utils import *
import importlib
from train_utils.checkpoint_helpers import get_best_checkpoint_details


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[2])

    ckpt_path = get_checkpoint_path(base_name, cfg.name)
    lmdb_path = get_lmdb_path()
    cfg.save_dir = f"{ckpt_path}/{base_name}/{cfg.name}"

    cfg.main_runner = "trainer.complete_translation_trainer"
    cfg.project_name = "isl_translation"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
    }

    base_bs = 8
    # VRAM-aware batch sizing (same as PHOENIX stage 2)
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    _ngpu = torch.cuda.device_count()
    if _vram_gb >= 70:
        cfg.bs = 8 * _ngpu
        cfg.accum = 1
    elif _vram_gb >= 40:
        cfg.bs = 6 * _ngpu
        cfg.accum = 1
    elif _vram_gb >= 22:
        cfg.bs = 4 * _ngpu
        cfg.accum = 2
    else:
        cfg.bs = 2 * _ngpu
        cfg.accum = 4
    cfg.num_workers = min(min(cfg.bs, int(10 * torch.cuda.device_count())), 10)

    cfg.gate_grad_multiplier = 1.0

    cfg.lr = 3e-4
    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict({
        "lr_scale_factor": 0.01,
        "num_cycles": 1,
        "start_value_mult": 0.7,
        "end_value_mult": 0.7,
        "warmup_epochs": 5,
    })

    cfg.optimizer_name = "adamw"
    cfg.optimizer_params = config_dict.ConfigDict({
        "lr": cfg.lr,
        "weight_decay": 0.001,
    })

    cfg.criterion_name = "losses.base_loss"
    cfg.criterion_params = config_dict.ConfigDict({
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "losses.loss_functions.ce_loss",
                "loss_params": {
                    "src_name": "logits",
                    "tgt_name": "gt_ids",
                    "mask_name": "gt_text_mask",
                    "label_smoothing": 0.1,
                },
                "weight": 1.0,
            },
        }
    })

    cfg.max_epochs = 60
    cfg.model_checkpoint_dir = ""

    cfg.train_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.valid_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.test_ds_name = "dataloaders.phoenix_video_dataset"
    train_ds_params = {
        "csv_dir": f"{code_path}/data/isl/ISL.train.corpus.csv",
        "pseudo_gloss_dir": f"{code_path}/data/isl/processed_words.isl_pkl",
        "sep": "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/isl/lmdb_videos",
            "isValid": False,
        },
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": True,
    }
    cfg.train_ds_params = config_dict.ConfigDict(train_ds_params)

    valid_ds_params = {
        "csv_dir": f"{code_path}/data/isl/ISL.dev.corpus.csv",
        "pseudo_gloss_dir": f"{code_path}/data/isl/processed_words.isl_pkl",
        "sep": "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/isl/lmdb_videos",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.valid_ds_params = config_dict.ConfigDict(valid_ds_params)

    test_ds_params = {
        "csv_dir": f"{code_path}/data/isl/ISL.test.corpus.csv",
        "pseudo_gloss_dir": f"{code_path}/data/isl/processed_words.isl_pkl",
        "sep": "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/isl/lmdb_videos",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.test_ds_params = config_dict.ConfigDict(test_ds_params)

    cfg.lm_name = "facebook/xglm-1.7B"  # XGLM supports English natively
    cfg.additional_tokens = {}
    cfg.pretext = ""
    cfg.replacement_pickle = None

    # Point to the ISL stage 1 config (not PHOENIX's)
    cfg.stage1_name = "configs.isl.isl_stage1_config"
    mod = importlib.import_module(cfg.stage1_name, package=None)
    stage1_config = mod.get_config()

    stage1_name = stage1_config["model_name"]
    stage1_params = stage1_config["model_params"].to_dict()
    stage1_ckpt_dir = stage1_config["save_dir"]
    stage1_ckpt = get_best_checkpoint_details(
        stage1_ckpt_dir, best_checkpoint_name="_result_checkpoint_"
    )[0]
    assert stage1_ckpt is not None and stage1_ckpt != ""

    cfg.model_name = "models.trial_models.test_stage2_model"
    cfg.model_params = {
        "stage1_name": stage1_name,
        "stage1_params": stage1_params,
        "stage1_ckpt": stage1_ckpt,
        "post_name": "models.post_models.linear_pos_head",
        "post_params": {"pos_type": "sine", "pre_pos": True},
        "llm_name": cfg.lm_name,
        "lang_backbone_name": "models.huggingface.modeling_xglm",
        "adaptor_params": {
            "adapt_layers": list(np.arange(0, 24, 1)),
            "lora_layers": list(np.arange(0, 24, 1)),
            "w_lora_ff": False,
            "lora_rank": 4,
            "lora_drop": 0.1,
            "gate_type": "clamp",
            "lora_a": 4.0,
            "adapt_tokens": False,
        },
        "freeze": False,
    }

    cfg.gen_params = {"max_length": 64, "temperature": 1.0, "num_beams": 4}

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.mixup = False
    cfg.logger_name = ["text"]
    cfg.resume = True
    cfg.train_length = None
    cfg.val_length = None
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/obleu"
    cfg.bfloat16_only = False
    cfg.apply_metric_splitter = False
    cfg.append_string = ""
    cfg.watch_grad = True
    return cfg
