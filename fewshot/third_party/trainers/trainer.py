import os
import sys
import functools
import numpy as np

import time
import math
import torch
from typing import Dict, List, Optional, Union, Any
from torch.utils.checkpoint import checkpoint_sequential
from tqdm.auto import tqdm
import warnings
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.nn.functional as F
import collections


from transformers import __version__
from transformers.configuration_utils import PretrainedConfig

from transformers import Trainer
from transformers.file_utils import (
    is_torch_tpu_available,
    is_sagemaker_mp_enabled,
    is_sagemaker_dp_enabled,
    is_apex_available,
    CONFIG_NAME
)
from transformers.deepspeed import deepspeed_init
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.utils import logging
from transformers.trainer_utils import (
    EvalLoopOutput,
    ShardedDDPOption,
    speed_metrics,
    TrainOutput,
    get_last_checkpoint,
    set_seed
)
from transformers.trainer_callback import (
    TrainerState,
)

from transformers.integrations import (
    is_fairscale_available,
    hp_params
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
    IterableDatasetShard
)
from transformers.optimization import (
    Adafactor,
    AdamW
)
from transformers.file_utils import WEIGHTS_NAME

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from fewshot.utils.utils import compute_accuracy_from_losses, get_aggregation 
from fewshot.third_party.models import RobertaForMaskedLM

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS


if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

from fewshot.utils.utils import get_aggregation, trim_input_ids, create_dir  

logger = logging.get_logger(__name__)


SOFT_MASK_LABELS = "extra_embeddings"


class BaseTrainer(Trainer):
    def __init__(self, eval_targets=None, task=None, metrics=None, extra_info=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_targets = eval_targets
        self.task = task
        self.metrics = metrics
        self.extra_info = extra_info 

    def evaluate(
        self,
        eval_datasets: Optional[Dataset] = None,
        eval_targets: Optional[Dataset] = None, 
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()
        output = self.eval_loop(
            eval_datasets = eval_datasets,
            eval_targets = eval_targets,
            description="Evaluation",
            metric_key_prefix=metric_key_prefix
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

 
    def eval_loop(
        self,
        eval_datasets,
        eval_targets,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Evaluation/Prediction loop."""
        logger.info(f"***** Running {description} *****")

        model = self._wrap_model(self.model, training=False)
        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_dataset
        eval_targets = eval_targets if eval_targets is not None else self.eval_targets 
        num_samples = eval_datasets[0].num_rows if isinstance(eval_datasets, list) else eval_datasets.num_rows
        logger.info(f"  Num examples = {num_samples}")

        model.eval()
        metrics = self.compute_pet_metrics(eval_datasets, model, self.extra_info[metric_key_prefix])

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    def _get_per_token_train_centroids_from_label_embeddings(self, model):
        centroids = {}
        start = 0
        num_masks = model.num_masks
        for label in range(self.model.config.num_labels):
            centroids[label] = model.extra_embeddings.weight.data[start:start+num_masks]
            start += num_masks 
        return centroids 

    def compute_pet_metrics(self, eval_datasets, model, extra_info):
        dataloader = self.get_eval_dataloader(eval_datasets)
        centroids=None
        if self.args.prototypical_eval:
            if self.args.label_embeddings_as_centroids:
                centroids = self._get_per_token_train_centroids_from_label_embeddings(model)
            else:
                centroids = self._compute_per_token_train_centroids(model)

        y_hats = []
        labels = []
        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                if self.args.train_classifier or self.args.classifier_eval:
                    logits = model(**inputs)["logits"]
                else:
                    logits = self.evaluate_pet(model, inputs, centroids=centroids)
                y_hat = torch.argmax(logits, axis=1).cpu().detach().numpy()
                y_hats.extend(y_hat) 
                labels.extend(inputs["labels"].cpu().detach().numpy())

        results = {}
        for metric in self.metrics:
            results.update(metric(y_hats, labels, extra_info))
        results["average"] = np.mean(list(results.values()))
        return results

    def evaluate_pet(self, model, batch, centroids=None):
        """Evaluates the model on the given inputs."""
        candidates_ids = batch["candidates_ids"]
        candidates_ids = candidates_ids.permute(1, 0, 2)
        num_labels = candidates_ids.shape[0]
        log_probs = []

        for label in range(num_labels):
            candidate_labels = candidates_ids[label]

            if self.args.soft_pet:
                if self.args.prototypical_eval:
                    log_prob = self._get_prototypical_candidate_eval_probability(model, batch, label, centroids)
                else:
                    log_prob = self._get_candidate_soft_log_probability_with_extra_tokens(model, batch, label, 
                         decoding_strategy=self.args.decoding_strategy)
            else:
                log_prob = self._get_candidate_log_probability(model, batch, candidate_labels[0],
                    decoding_strategy=self.args.decoding_strategy)
            log_probs.append(log_prob)
        
        result = torch.tensor([log_probs])
        
        if self.args.prototypical_eval:
            result = result.squeeze()
            result = result.permute(1, 0)
        return result 

    def get_masks_embeds(self, model, batch):
        """Returns mask embeddings of size batch_size x num_masks x hidden_dim"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        if self.args.prompt_tune:
            input_ids, attention_mask, inputs_embeds = model.append_prompts(input_ids, attention_mask, inputs_embeds=None)
            hidden_states = model.roberta(input_ids=None, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        else:
            hidden_states = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = hidden_states[0]
        batch_size = input_ids.shape[0]
        mask_indices = (input_ids == model.config.mask_token_id).nonzero()[:, -1].view(batch_size, -1)
        return hidden_states[torch.arange(hidden_states.shape[0]).unsqueeze(-1), mask_indices]

    def _compute_per_token_train_centroids(self, model):
        """For training datapoints belonging to each label, computes the average embedding of masked tokens
        across all samples of each label. 
        Returns a dictionary from labels to embedding size of shape [num_tokens, hidden_dim]"""
        def get_label_samples(dataset, label):
            return dataset.filter(lambda example: int(example['labels']) == label)
        label_to_token_centroids = {}
        for label in range(self.model.config.num_labels):
            data = get_label_samples(self.train_dataset, label)
            dataloader = self.get_eval_dataloader(data)
            mask_embeds = [] 
            for _, inputs in enumerate(dataloader):
                batch = self._prepare_inputs(inputs)
                with torch.no_grad():
                    mask_embeds.append(self.get_masks_embeds(model, batch))
            # Computes the averaged mask embeddings for the samples of this label.
            label_to_token_centroids[label] = torch.mean(torch.cat(mask_embeds, dim=0), dim=0)
        return label_to_token_centroids

    def _get_prototypical_candidate_eval_probability(self, model, batch, label, centroids):
        def cosine_similarity(embed1, embed2):
            embed1 = F.normalize(embed1, dim=-1)
            embed2 = F.normalize(embed2, dim=-1)
            return F.cosine_similarity(embed1, embed2, dim=2)
        def euclidean_similarity(embed1, embed2):
            embed1 = F.normalize(embed1, dim=-1)
            embed2 = F.normalize(embed2, dim=-1)
            return torch.exp(-(embed1 - embed2).pow(2).sum(-1))
            
        mask_embeds = self.get_masks_embeds(model, batch)  # batch_size x num_masks x hidden_dim  
        label_centroids = centroids[label][None, :]        # 1 x num_masks x hidden_dim  
        if self.args.prototypical_similarity == "cos":
            similarity = cosine_similarity(label_centroids, mask_embeds) # batch_size x num_masks 
        elif self.args.prototypical_similarity == "euc":
            similarity = euclidean_similarity(label_centroids, mask_embeds) # batch_size x num_masks    
        aggregate = get_aggregation(self.args.eval_soft_pet_aggregation)
        prob = aggregate(similarity, dim=-1)
        if self.args.eval_soft_pet_aggregation in ["min", "max"]:
            prob = prob[0]
        return prob.cpu().detach().numpy().tolist()

    def get_masks_probs(self, model, batch, prev_mask_ids):
        assert batch["input_ids"].shape[0] == 1, "we only support batch size of 1 during eval."
        input_ids = trim_input_ids(
            batch["input_ids"],
            num_masks = self.args.num_extra_tokens,
            pad_token_id = self.model.config.pad_token_id,
            mask_token_id = self.model.config.mask_token_id    
        ) 
        masks_positions = [idx for idx, tok_id in enumerate(input_ids[0, :]) if
            tok_id == self.model.config.mask_token_id]
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)

        for i, id in enumerate(prev_mask_ids):
            inputs_embeds[0, masks_positions[i], :] =\
            self.model.roberta.embeddings.word_embeddings(torch.tensor([id]).cuda())

        outputs = model(input_ids=None, inputs_embeds=inputs_embeds)
        next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]
        # We find the next mask position.
        prob = next_token_logits[masks_positions[len(prev_mask_ids)]]
        return prob

    def _get_candidate_soft_log_probability_with_extra_tokens(self, model, batch, label, decoding_strategy="default"):
        """Computes the probability of the given candidate labels."""
        num_masks = self.model.num_masks
        assert batch["input_ids"].shape[0] == 1, "we only support batch size of 1 during eval."
        # removes the pad and keeps at most the num_mask tokens of masks.
        input_ids = trim_input_ids(
            batch["input_ids"],
            num_masks = num_masks,
            pad_token_id = self.model.config.pad_token_id,
            mask_token_id = self.model.config.mask_token_id    
        )
        masks_positions = [idx for idx, tok_id in enumerate(input_ids[0, :]) if
                           tok_id == self.model.config.mask_token_id]
        mask_labels = self.model.map_labels_to_mask_ids(torch.tensor([label]).cuda())
        mask_labels = mask_labels.detach().cpu().numpy().tolist()
        if not isinstance(mask_labels, list):
            mask_labels = [mask_labels]
        # first element is the index in the sequence, second is the position within all masks.
        masks_positions = [(mask_position, mask_label) for mask_position, mask_label in zip(masks_positions, mask_labels)]
        log_probabilities = []
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        for i in range(num_masks):
            outputs = model(input_ids=None, inputs_embeds=inputs_embeds)
            next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]
            if decoding_strategy == "parallel":
                for m_pos, m_id in masks_positions:
                    log_probabilities.append(math.log(next_token_logits[m_pos][m_id].item()))
                break
            mask_pos, masked_id = None, None
            max_prob = None
            for m_pos, m_id in masks_positions:
                m_prob = next_token_logits[m_pos][m_id].item()
                if max_prob is None or m_prob > max_prob:
                    max_prob = m_prob
                    mask_pos, masked_id = m_pos, m_id
            log_probabilities.append(math.log(max(max_prob, sys.float_info.min)))
            # put the mask position with maximum probability in its place.
            shift = 0 
            inputs_embeds[0, mask_pos, :] = self.model.extra_embeddings(torch.tensor([masked_id-shift]).cuda()) #self.model.soft_mask_labels[label][min_n, :]
            masks_positions.remove((mask_pos, masked_id))
        return sum(log_probabilities)

    def _get_candidate_log_probability(self, model, batch, 
        candidate_labels, decoding_strategy="default"):
        """Computes the probability of the given candidate labels."""  
        num_masks = sum(1 for token_id in candidate_labels if token_id != -100)
        # removes the pad and keeps at most the num_mask tokens of masks.
        input_ids = trim_input_ids(
            batch["input_ids"],
            num_masks = num_masks,
            pad_token_id = self.model.config.pad_token_id,
            mask_token_id = self.model.config.mask_token_id    
        )
        log_probabilities = []
        while True:
            masks = [(idx, tok_id) for idx, tok_id in enumerate(candidate_labels) if tok_id != -100]
            if not masks:  # there are no masks left to process, we are done
                break
            outputs = model(input_ids)
            next_token_logits = torch.nn.Softmax(dim=2)(outputs[0])[0]
            if decoding_strategy == "ltr":
                mask_pos, masked_id = masks[0]
                max_prob = next_token_logits[mask_pos][masked_id].item()
            elif decoding_strategy == "parallel":
                for m_pos, m_id in masks:
                    log_probabilities.append(math.log(next_token_logits[m_pos][m_id].item()))
                break
            else:
                mask_pos, masked_id = None, None
                max_prob = None
                for m_pos, m_id in masks:
                    m_prob = next_token_logits[m_pos][m_id].item()
                    if max_prob is None or m_prob > max_prob:
                        max_prob = m_prob
                        mask_pos, masked_id = m_pos, m_id
            log_probabilities.append(math.log(max_prob))
            # put the mask position with maximum probability in its place.
            input_ids[0][mask_pos] = masked_id
            candidate_labels[mask_pos] = -100
        return sum(log_probabilities)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
           {
                    "params": [p for n, p in self.model.named_parameters() if SOFT_MASK_LABELS in n],
                    "lr": self.args.soft_mask_labels_learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters()\
                               if n in decay_parameters and\
                                  SOFT_MASK_LABELS not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters()\
                               if n not in decay_parameters and\
                                SOFT_MASK_LABELS not in n],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer