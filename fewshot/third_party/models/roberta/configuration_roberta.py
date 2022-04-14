"""RoBERTa configuration."""

from transformers.models.roberta import RobertaConfig 

class RobertaConfig(RobertaConfig):
    def __init__(self,
        soft_pet=False,
        extra_tokens_init="tokens",
        model_name_or_path=None,
        train_in_batch=False,
        extra_embd_initializer_range=0.02,
        token_hinge_loss=False,
        multiclass_ce_loss=False,
        prompt_tune=False,
        prompt_length=20,
        init_prompt_from_vocab=True,
        prompt_init_range=1e-4,
        eval_soft_pet_aggregation=None,
        soft_pet_aggregation=None,
        prototypical_similarity="cos",
        **kwargs):
        super().__init__(**kwargs)
        self.soft_pet = soft_pet
        self.extra_tokens_init = extra_tokens_init
        self.model_name_or_path=model_name_or_path
        self.train_in_batch=train_in_batch
        self.extra_embd_initializer_range=extra_embd_initializer_range
        self.token_hinge_loss = token_hinge_loss
        self.multiclass_ce_loss = multiclass_ce_loss
        self.prompt_tune = prompt_tune 
        self.prompt_length = prompt_length
        self.init_prompt_from_vocab = init_prompt_from_vocab
        self.prompt_init_range = prompt_init_range
        self.eval_soft_pet_aggregation = eval_soft_pet_aggregation
        self.soft_pet_aggregation = soft_pet_aggregation
        self.prototypical_similarity = prototypical_similarity