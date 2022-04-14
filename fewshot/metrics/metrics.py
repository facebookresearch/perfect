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
