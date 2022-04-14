"""Defines different way of processing the data."""
import sys 
import torch
from typing import List, Tuple


class MLMProcessor(torch.nn.Module):
    """Process the data for a model which is pretrained with the masked
    language modeling loss."""
    def __init__(self, tokenizer, tokenized_verbalizers, max_seq_length, 
        processor, mask_length=None, train_classifier=False):
        super(MLMProcessor, self).__init__()
        self.tokenizer = tokenizer 
        self.tokenized_verbalizers = tokenized_verbalizers
        self.max_seq_length = max_seq_length
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.processor = processor 
        # In case of using soft_pet, we path the mask_length so we use the one passed.
        self.mask_length = mask_length
        self.train_classifier = train_classifier 

    # copied from pet 
    def seq_length(self, parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    # copied from pet 
    def remove_last(self, parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    # copied from pet 
    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]]):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self.seq_length(parts_a) + self.seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - self.max_seq_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self.seq_length(parts_a, only_shortenable=True) > self.seq_length(parts_b, only_shortenable=True):
                self.remove_last(parts_a)
            else:
                self.remove_last(parts_b)

    def tokenize(self, text_list):
        """Gets a list of Text enteries and tokenize them, returns the output as a list of tuples with the
        tokenized text and the shortenable entry."""
        return [(self.tokenizer.encode(text.text, add_special_tokens=False), text.shortenable) for text in text_list]

    def get_tokens(self, tuple_list):
        if not tuple_list:
            return None 
        return [token_id for part, _ in tuple_list for token_id in part]

    def prepare_classification_inputs(self, example):
        part_0, part_1 = self.processor.get_classification_parts(
            example=example
        )
        # Roberta does not use token_type ids.
        input_ids = self.tokenizer.encode_plus(
                part_0,
                part_1,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                truncation=True,
        )["input_ids"]
        target = self.processor.get_target(example=example)
        attention_mask = [1] * len(input_ids)
        n_mask = self.max_seq_length - len(input_ids)
        # Pads the tokens and attention mask.
        input_ids = input_ids + [self.pad_token_id] * n_mask
        attention_mask = attention_mask + [0] * n_mask
        extra_fields = self.processor.get_extra_fields(example=example)
        return {
            'labels': int(target),
            'attention_mask': attention_mask,
            'input_ids': input_ids,
            'extra_fields': extra_fields
        }


    def forward(self, example):
        if self.train_classifier:
            return self.prepare_classification_inputs(example)

        # For PET, we assume we have only one verbalizer per label.
        # We need not to cache it, and use the given one only if this is not None.
        tokenized_verbalizers = self.tokenized_verbalizers
        if tokenized_verbalizers is None:
            tokenized_verbalizers = self.processor.get_tokenized_verbalizers(example=example)
        
        if self.mask_length is None:
            mask_length = max([len(t[0]) for t in tokenized_verbalizers])
        else:
            mask_length = self.mask_length

        part_0, part_1 = self.processor.get_sentence_parts(
            example=example, 
            mask_length=mask_length
        )

        target = self.processor.get_target(example=example)
        part_0_tuples = self.tokenize(part_0)
        part_1_tuples = self.tokenize(part_1)
        self.truncate(part_0_tuples, part_1_tuples)
        token_ids_0 = self.get_tokens(part_0_tuples) 
        token_ids_1 = self.get_tokens(part_1_tuples)        
        
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            token_ids_0=token_ids_0, 
            token_ids_1=token_ids_1)
        attention_mask = [1] * len(input_ids)
        n_mask = self.max_seq_length - len(input_ids)

        # Pads the tokens and attention mask.
        input_ids = input_ids + [self.pad_token_id] * n_mask
        attention_mask = attention_mask + [0] * n_mask

        # Builds candidates tokens ids.
        candidates_ids = []
        mask_start = input_ids.index(self.mask_token_id)
        for verbalizers_list in tokenized_verbalizers:
            tokens = verbalizers_list[0]
            candidate_ids = [-100]*len(input_ids)
            mask_end = mask_start + len(tokens)
            candidate_ids[mask_start:mask_end] = tokens
            candidates_ids.append(candidate_ids)

        extra_fields = self.processor.get_extra_fields(example=example)

        return {
            'candidates_ids': candidates_ids,
            'labels': int(target),
            'attention_mask': attention_mask,
            'input_ids': input_ids,
            'extra_fields': extra_fields
        }
