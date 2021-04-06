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

# Automatically import models. This will update Transformer's model mappings so that
# custom models can be loaded via AutoModelForMaskedLM and related auto-constructors.
import models

from .base import CONFIGS as BASE
from .bert_replication import CONFIGS as BERT_REPLICATION
from .bertitos import CONFIGS as BERTITOS
from .distillation import CONFIGS as DISTILLATION
from .finetuning import CONFIGS as FINETUNING
from .hpsearch import CONFIGS as HPSEARCH
from .one_cycle_lr import CONFIGS as ONE_CYCLE_LR
from .rigl_bert import CONFIGS as RIGL_BERT
from .sparse_bert import CONFIGS as SPARSE_BERT

"""
Import and collect all experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(BERT_REPLICATION)
CONFIGS.update(BERTITOS)
CONFIGS.update(DISTILLATION)
CONFIGS.update(FINETUNING)
CONFIGS.update(HPSEARCH)
CONFIGS.update(ONE_CYCLE_LR)
CONFIGS.update(RIGL_BERT)
CONFIGS.update(SPARSE_BERT)