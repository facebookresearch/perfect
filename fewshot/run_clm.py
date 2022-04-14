#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import torch 
import logging
import numpy as np 
import os
import sys
import functools

os.environ["WANDB_DISABLED"] = "true"

import datasets

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from fewshot.third_party.models import (
    RobertaForMaskedLM,
    RobertaConfig,
    RobertaForSequenceClassification
)
from fewshot.third_party.trainers import BaseTrainer
from fewshot.utils.utils import (
    load_json,
    get_adapter_config, 
    set_trainable_params_for_adapters,
    set_config_args,
    freeze_model,
    set_layernorms_trainable_params,
    set_trainable_params_for_bitfit,
    set_trainable_params_for_prompt_tuning
)
from fewshot.training_args import ModelArguments, DataTrainingArguments, FewShotTrainingArguments, AdapterArguments
from fewshot.data.preprocessing import  MLMProcessor
from fewshot.data.tasks import AutoTask
from fewshot.data.processors import AutoProcessor

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FewShotTrainingArguments, AdapterArguments))
    print(sys.argv[1])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
   
    if training_args.classifier_eval or training_args.prototypical_eval:
        assert training_args.classifier_eval != training_args.prototypical_eval

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    task = AutoTask.get(
        task=data_args.task, 
        data_seed=data_args.data_seed, 
        num_samples=data_args.K,
        cache_dir=model_args.cache_dir, 
        data_dir=data_args.data_dir)
    raw_datasets = task.get_datasets()
 
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    adapter_config = get_adapter_config(adapter_args)
    processor = AutoProcessor.get(
        task=data_args.task,
        tokenizer=tokenizer,
        with_pattern=not training_args.soft_pet and not data_args.no_pattern,
        pattern_id=data_args.pattern_id,
        mask_position=training_args.mask_position
    )
   
    verbalizers = {}
    # These tokenized verbalizers are used to inialize the embedding matrix for the extra tokens
    # added for the label embedding. If we do not have a fixed set of verbalizers, we intialize 
    # this with an empty list, in this case the initialzation from verbalizers and random would be the same.
    if len(processor.get_verbalizers()) != 0:
        verbalizers["init"] = processor.get_tokenized_verbalizers()
    else:
        verbalizers["init"] = []


    # In case of PET, if this is possible to train it in batch, do it.
    if not training_args.soft_pet and training_args.vectorize_pet:
        # Test if verbalizers are of the same length.
        verbalizers = processor.get_verbalizers()
        if len(verbalizers) != 0 :
            verbalizers_tokens = processor.get_tokenized_verbalizers()
            lengths = [len(v[0]) for v in verbalizers_tokens]
            if len(np.unique(lengths)) == 1:
                training_args.per_device_train_batch_size = training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps
                training_args.gradient_accumulation_steps = 1
                training_args.train_in_batch = True 

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = RobertaConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = RobertaConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = RobertaConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    set_config_args(config, training_args)
    config.num_labels = task.num_labels
    config.model_name_or_path = model_args.model_name_or_path

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    if training_args.num_extra_tokens == -1:
        # If dataset, has static verbalizers, we can compute the length from the tokenized verbalizers' length.
        if len(processor.get_verbalizers()) != 0:
            training_args.num_extra_tokens = max([len(t[0]) for t in processor.get_tokenized_verbalizers()])
        else:
            # when dataset has dynamic verbalizers, we need to go through the whole training examples,
            # and compute the maximum length of the verbalizers.
            def get_max_num_verbalizers_tokens(example, processor):
                max_verbalizer_length = max([len(t[0]) for t in processor.get_tokenized_verbalizers(example)])
                return {"length": max_verbalizer_length}
            max_lengths = raw_datasets["train"].map(
                functools.partial(get_max_num_verbalizers_tokens,  processor=processor),
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False, 
                desc="Finding the length of max verbalizer in the training set",
            )
            training_args.num_extra_tokens = max(max_lengths["length"])
           
    if training_args.soft_pet:
        start = config.vocab_size
        extra_token_verbalizers = []
        for l in range(config.num_labels):
            tokens = [i for i in range(start, start+training_args.num_extra_tokens)]
            start += training_args.num_extra_tokens
            extra_token_verbalizers.append([tokens])
        verbalizers["extra"] = extra_token_verbalizers

    processor = MLMProcessor(
        tokenizer=tokenizer,
        tokenized_verbalizers=None if not training_args.soft_pet else extra_token_verbalizers,
        max_seq_length=data_args.max_seq_length,
        processor=processor,
        mask_length = training_args.num_extra_tokens if training_args.soft_pet else None,
        train_classifier = training_args.train_classifier  
    )
    config.mask_token_id = tokenizer.mask_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # In case of loading from initial verbalizers, we pass the initialization to the model.
    def get_initial_weights(config, verbalizers):
        num_masks = len(verbalizers["extra"][0][0])
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        num_extra_tokens = num_masks * config.num_labels
        weight_init = torch.normal(mean=0.0, std=config.extra_embd_initializer_range, size=(num_extra_tokens, config.hidden_size))
        # initialize from the given verbalizers.
        # This is to initialize with this, in case there is no more verbalizers tokens.
        start = 0
        for _, v in enumerate(verbalizers["init"]):
            tokens = np.array(v[0])[:num_masks]
            weight_init.data[start:start+len(tokens)] = \
                model.get_input_embeddings().weight[torch.tensor(tokens)].clone().detach()
            start += num_masks
        return weight_init 

    
    
    # Computes the weights for the extra tokens.
    extra_embedding_weight = get_initial_weights(config, verbalizers) if training_args.extra_tokens_init == 'verbalizers' else None 
    
    # TODO: write an automodel class here.
    if model_args.model_name_or_path:
        # TODO: for now tokenizers are not used, but we need to think if later 
        # we want to use them to pad or not.
        if training_args.train_classifier:
            model = RobertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None
            )
        else:
            model = RobertaForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                adapter_config=adapter_config,
                tokenized_verbalizers=verbalizers,
                extra_embedding_weight = extra_embedding_weight
            )
    else:
        model = RobertaForMaskedLM.from_config(config) 
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    model.resize_token_embeddings(len(tokenizer))


    # In case of prompt_tune, updates the embedding.
    if training_args.prompt_tune:
        assert training_args.soft_pet == True and training_args.prototypical_eval, "currently this is only implemented with soft_pet and prototypical_eval"
        model.create_prompt_embedding()

    # freeze parameters.
    if adapter_args.adapter_tune:
       set_trainable_params_for_adapters(model, adapter_args.tune_layernorms)
    if adapter_args.freeze_model:
       freeze_model(model)
    if adapter_args.tune_layernorms:
       set_layernorms_trainable_params(model, adapter_args.tune_layernorms)
    if adapter_args.tune_biases:
       set_trainable_params_for_bitfit(model, adapter_args.tune_lm_head)
    if training_args.prompt_tune:
        set_trainable_params_for_prompt_tuning(model)
    if adapter_args.print_params:  
       total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       total_params = sum(p.numel() for p in model.parameters()) 
       print("Total params ", total_params, " total trainable params ", total_trainable_params)
       for n, p in model.named_parameters():
         if p.requires_grad:
            print(n)
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    def extract_targets(examples):
        targets = examples["label"]
        targets = [int(target) for target in targets]
        return {"targets": targets}   
    
    if training_args.do_train:
        if "train" not in raw_datasets: 
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
            processor,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False, #not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
            )


    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_targets = eval_dataset.map(
            extract_targets,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on eval dataset",    
            )
            eval_dataset = eval_dataset.map(
                    processor,
                    batched=False,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=False, #not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_targets = predict_dataset.map(
            extract_targets,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on predict dataset",    
            )
            predict_dataset = predict_dataset.map(
                processor,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file= False, #not data_args.overwrite_cache,
                desc="Running tokenizer on predict dataset",
            )

    data_collator = default_data_collator
    all_datasets = {"train": train_dataset, "eval": eval_dataset, "predict": predict_dataset}
    extra_info = {k: v["extra_fields"] for k, v in all_datasets.items()} 
    train_dataset = train_dataset.remove_columns("extra_fields")
    eval_dataset = eval_dataset.remove_columns("extra_fields")
    predict_dataset = predict_dataset.remove_columns("extra_fields")

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        eval_targets=eval_targets,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        task=data_args.task,
        metrics=task.metric,
        extra_info = extra_info,
    )

    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       trainer.save_metrics("arguments", load_json(sys.argv[1]))

    # Training
    performance_metrics = {}
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"training time(min)": total_time})
        
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        performance_metrics.update({"peak_memory(GB)": peak_memory})
    
    if training_args.compute_memory or training_args.compute_time and not training_args.compute_inference_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        eval_samples = eval_dataset[0].num_rows if isinstance(eval_dataset, list) else eval_dataset.num_rows
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else eval_samples
        metrics["eval_samples"] = min(max_eval_samples, eval_samples)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        if training_args.compute_inference_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        metrics = trainer.evaluate(
            eval_datasets=predict_dataset, 
            eval_targets=predict_targets, 
            metric_key_prefix="predict"
        )

        if training_args.compute_inference_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"inference time(min)": total_time})

        predict_samples = predict_dataset[0].num_rows if isinstance(predict_dataset, list) else predict_dataset.num_rows
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else predict_samples
        metrics["predict_samples"] = min(max_predict_samples, predict_samples)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    if training_args.compute_memory or training_args.compute_time or training_args.compute_inference_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
