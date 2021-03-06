# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
from copy import deepcopy

from transformers import Trainer

from callbacks import RezeroWeightsCallback
from trainer_mixins import (
    DistillationTrainerMixin,
    LRRangeTestMixin,
    OneCycleLRMixin,
    RigLMixin,
)

from .finetuning import finetuning_bert100k_glue_simple, finetuning_bert700k_glue
from .sparse_bert import fully_static_sparse_bert_100k_fp16
from .sparse_bertitos import small_bert_sparse_100k, tiny_bert_sparse_100k


"""
These are configs that combine three strong approaches to sparse training
    1. Knowledge Distillation
    2. RigL Dynamic Sparsity
    3. One Cycle Learning Rate
"""


class KDRigLOneCycleLRTrainer(OneCycleLRMixin,
                              DistillationTrainerMixin,
                              RigLMixin,
                              Trainer):
    pass


class KDLRRangeTestTrainer(LRRangeTestMixin,
                           DistillationTrainerMixin,
                           Trainer):
    pass


# ---------
# Tiny BERT
# ---------

# This combines KD + RigL + OneCycle LR on Tiny BERT.
tiny_bert_trifecta_300k = deepcopy(tiny_bert_sparse_100k)
tiny_bert_trifecta_300k.update(
    max_steps=300000,
    model_type="fully_static_sparse_bert",
    overwrite_output_dir=True,

    # Sparsity callback
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,

    trainer_class=KDRigLOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.0075,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],

        # RigL
        prune_fraction=0.3,
        prune_freq=100,
    ),
)

tiny_bert_trifecta_100k = deepcopy(tiny_bert_trifecta_300k)
tiny_bert_trifecta_100k.update(
    max_steps=100000,
)


# This fine-tunes a pretrained model from `tiny_bert_trifecta_100k`
finetuning_tiny_bert_trifecta_100k = deepcopy(finetuning_bert700k_glue)
finetuning_tiny_bert_trifecta_100k.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/tiny_bert_trifecta_100k",  # noqa: E501
)


# LR Range Test for training with KD and OneCycle LR. It's assumed the observed max_lr
# will carry over to training with RigL.
tiny_bert_trifecta_lr_range_test = deepcopy(tiny_bert_trifecta_300k)
tiny_bert_trifecta_lr_range_test.update(
    max_steps=100,
    trainer_class=KDLRRangeTestTrainer,
    # eval_steps=1,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.05,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
    overwrite_output_dir=True,
    do_eval=True,
)

# ---------
# Small BERT
# ---------

# This combines KD + RigL + OneCycle LR on Small BERT.
# This gets a NaN eval-loss for max_lr=0.006
small_bert_trifecta_300k = deepcopy(small_bert_sparse_100k)
small_bert_trifecta_300k.update(
    max_steps=300000,
    model_type="fully_static_sparse_bert",
    overwrite_output_dir=True,

    # Using batch_size of 16 instead of 128 since we're training on 8 GPUs.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    # Sparsity callback
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,

    trainer_class=KDRigLOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.003,
        pct_start=0.10,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],

        # RigL
        prune_fraction=0.3,
        prune_freq=100,
    ),
)


small_bert_trifecta_100k = deepcopy(small_bert_trifecta_300k)
small_bert_trifecta_100k.update(
    max_steps=100000,
)


# LR Range Test for training with KD and OneCycle LR. It's assumed the observed max_lr
# will carry over to training with RigL.
small_bert_trifecta_lr_range_test = deepcopy(small_bert_trifecta_300k)
small_bert_trifecta_lr_range_test.update(
    max_steps=100,
    trainer_class=KDLRRangeTestTrainer,

    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.05,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
    overwrite_output_dir=True,
    do_eval=True,
)


# ---------
# BERT Base
# ---------


# BERT Base with KD + RigL + OneCycle LR
# This achieves and eval-loss of 2.138, just slightly under 2.154 from its dense
# counterpart. See `sparse_v5_trifecta_100k` in the README for more details.
# This took 18h 21m on four p3dn.24xlarges.
bert_sparse_trifecta_100k = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_trifecta_100k.update(
    trainer_class=KDRigLOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.0012,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],

        # RigL
        prune_fraction=0.3,
        prune_freq=100,
    ),
    overwrite_output_dir=True,
)


# This fine-tunes a pretrained model from `bert_sparse_100k_kd_oncycle_lr` above.
finetuning_bert_sparse_trifecta_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert_sparse_trifecta_100k_glue.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_trifecta_100k",  # noqa: E501
)

finetuning_bert_sparse_trifecta_100k_glue_simple = deepcopy(
    finetuning_bert100k_glue_simple)
finetuning_bert_sparse_trifecta_100k_glue_simple.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/"
    "bert_sparse_80%_trifecta_100k",
)

# alias with a shorter variable name for pep8 compliance below
ft_bert_sp_tri_100k_g_s = finetuning_bert_sparse_trifecta_100k_glue_simple


CONFIGS = dict(
    # Tiny BERT
    tiny_bert_trifecta_100k=tiny_bert_trifecta_100k,
    tiny_bert_trifecta_300k=tiny_bert_trifecta_300k,
    tiny_bert_trifecta_lr_range_test=tiny_bert_trifecta_lr_range_test,
    finetuning_tiny_bert_trifecta_100k=finetuning_tiny_bert_trifecta_100k,

    # Small BERT
    small_bert_trifecta_100k=small_bert_trifecta_100k,
    small_bert_trifecta_300k=small_bert_trifecta_300k,
    small_bert_trifecta_lr_range_test=small_bert_trifecta_lr_range_test,

    # BERT Base
    bert_sparse_trifecta_100k=bert_sparse_trifecta_100k,
    finetuning_bert_sparse_trifecta_100k_glue=finetuning_bert_sparse_trifecta_100k_glue,
    finetuning_bert_sparse_trifecta_100k_glue_simple=ft_bert_sp_tri_100k_g_s,

)
