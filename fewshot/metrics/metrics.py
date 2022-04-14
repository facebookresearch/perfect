# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# This file is part of PERFECT.
# See https://github.com/facebookresearch/perfect for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict


def accuracy(predictions, targets, extra_info=None) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}


def exact_match(predictions, targets):
  """Computes whether the targets match predictions exactly."""
  return {"em": 100 * float(np.array_equal(targets, predictions))}


# This is copied from pet.
def group_exact_match(predictions, targets, extra_info):
    """Computes the average exact match(EM) score for predictions and targets 
    corresponding to each question id."""
    question_ids = [v["group"] for v in extra_info]
    unique_q_ids = set(question_ids)
    id_to_targets = defaultdict(list)
    id_to_predictions = defaultdict(list)
    for q_id, target, prediction in zip(question_ids, targets, predictions):
        id_to_targets[q_id].append(target)
        id_to_predictions[q_id].append(prediction)

    # Computing the average em score for over question ids.
    ems = []
    for q_id in question_ids:
        ems.append(exact_match(id_to_predictions[q_id], id_to_targets[q_id])["em"])
    return {"em": np.mean(ems)}


def f1_macro(predictions, targets, extra_info=None):
    return {"f1-macro": 100*f1_score(targets, predictions, average="macro")}


def f1(predictions, targets, extra_info=None):
    return {"f1": 100*f1_score(targets, predictions)}
