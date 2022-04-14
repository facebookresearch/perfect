from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class FewShotTrainingArguments(TrainingArguments):
    # Prompt-tuning parameters.
    prompt_tune: Optional[bool] = field(default=False, metadata={"help": "If sets, adds prompts token to the input and only tune them."}) 
    prompt_length: Optional[int] = field(default=20, metadata={"help": "Sets the number of tokens for prompt-tuning."})
    init_prompt_from_vocab: Optional[bool] = field(default=True, metadata={"help": "If set, initializes the prompt tokens' embedding"
        "from the given pretrained model's vocabulary."})
    prompt_init_range: Optional[float] = field(default=1e-4, metadata={"help": "Defines the initialization range."})
    # perfect parameters.
    label_embeddings_as_centroids: Optional[bool]= field(default=False, metadata={"help": "if set, uses label embeddings as centroids."})
    mask_position: Optional[str] = field(default=None, metadata={"help": "This defines the position of mask in case of"
    "having two sentences `0`: [p,h,m],[]  `1`: [p,m,h],[]  `2`: [p],[m,h] , `3`: [p],[h,m]"})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set, computes the training time."})
    compute_inference_time: Optional[bool] = field(default=False, metadata={"help": "If set, computes the inference time."})
    compute_memory: Optional[bool] = field(default=False, metadata={"help": "If set, computes the memory."})
    train_classifier: Optional[bool] = field(default=False, metadata={"help": "If set trains a classifier in a conventional way of finetuning."})
    vectorize_pet: Optional[bool] = field(default=False, 
        metadata={"help": "If set and feasible (length of verbalizers are the same), vectorizes the pet."})
    multiclass_ce_loss: Optional[bool] = field(default=False,
        metadata={"help":"If set uses the multiclass cross-entropy loss."})
    token_hinge_loss: Optional[bool] = field(default=False, 
        metadata={"help": "If set, computes a multi-class classification hinge loss over the tokens."})
    classifier_eval: Optional[bool] = field(default=False, metadata={"help": "If set, uses the trained classifier" 
        "logits during evaluation."})
    prototypical_eval: Optional[bool] = field(default=False,
        metadata={"help": "If set, uses the prototypical evaluation during the inference."})
    prototypical_similarity: Optional[str] = field(default="cos",
        metadata={"help": "This can be `cos` for cosine similarity or `euc` for euclidean one."})
    extra_embd_initializer_range: Optional[float] = field(default=0.02,
        metadata={"help": "Defines the intialization range for the extra embedding added."}
    )
    train_in_batch: Optional[bool] = field(default=False,
        metadata={"help": "If set, trains the model in batches."} )
    decoding_strategy: Optional[str] = field(default="default", 
       metadata={"help": "This can be `default` or `parallel`: to feed in the input"
      "with masks only once to the encoder."})
    soft_mask_labels_learning_rate: Optional[float] = field(default=1e-5)
    eval_soft_pet_aggregation: Optional[str] = field(
        default=None, metadata={"help": "defines aggregation for eval."}
    )
    soft_pet_aggregation: Optional[str] = field(
       default=None, metadata={"help": "defines the aggregation operation for the losses."}
    )
    extra_tokens_init: Optional[str] = field(
       default="tokens", metadata={"help": "Defines the initialization for label embeddings."
       "`tokens`: initialize from random tokens, `random`: initialize randomly, `verbalizers`:" 
       "initialize from verbalizers."}
    )
    num_extra_tokens: Optional[int] = field(
        default=-1, metadata={"help": "Defines the number of mask tokens added in perfect, in" 
            "case of -1, it is computed from the length of verbalizers."}
    )
    soft_pet: Optional[bool] = field(
        default=False, metadata={"help": "If set, uses perfect model by computing the loss of the PET in the soft way"
        "by minimizing the embeddings of the tokens."}        
    )
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from gpt2, gpt2-large, gpt2-medium"} # + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
                    "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    add_masks: Optional[bool] = field(default=True, metadata={"help": "If set, adds mask tokens to the input."} )
    no_pattern: Optional[bool] = field(default=False, metadata={"help": "If set, removes the patterns."})
    data_dir: Optional[str] = field(default=None, metadata={"help": "Specifies the data directory."})
    data_seed: Optional[int] = field(default=100, metadata={"help": "Specifies the seed used to sample the data."})
    K: Optional[int] = field(default=16, metadata={"help": "Specifies the number of training samples."})
    task: Optional[str] = field(
        default=None, metadata={"help": "In case of passing the training files, it is additionally\
        required to pass the name of the task."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "Maximum input sequence length with taking into account special tokens."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pattern_id: Optional[int] = field(
        default=0, metadata={"help": "Defines a zero-based pattern index from the four available pattern."})
    

@dataclass
class AdapterArguments:
    """
    Arguments related to tuning a language model better for better generalization,
    including training with adapters, and pruning methods.
    """
    # Adapter's arguments.
    adapter_tune: Optional[bool] = field(default=False,
                                         metadata={"help": "If set to true, tune adapters."})
    add_layer_norm_before_adapter: Optional[bool] = field(default=False)
    add_layer_norm_after_adapter: Optional[bool] = field(default=False)
    nonlinearity: Optional[str] = field(default="gelu_new")
    reduction_factor: Optional[int] = field(default=16)
    tune_layernorms: Optional[bool] = field(default=False,
                                            metadata={"help": "If set, tunes the layernorms."})
    add_adapter_after_attention: Optional[bool] = field(default=True)
    add_adapter_after_feedforward: Optional[bool] = field(default=True)
    print_params: Optional[bool] = field(default=False,
                                         metadata={"help": "If set, prints all the parameters."})
    freeze_model: Optional[bool] = field(default=False,
                                         metadata={"help": "If set, freezes the model."})
    tune_biases: Optional[bool] = field(default=False,
                                        metadata={"help": "If set, tunes only biases."})
    tune_lm_head: Optional[bool] = field(default=False,
                                         metadata={"help": "If set, tunes the lm-head also when tuning biases."})    
