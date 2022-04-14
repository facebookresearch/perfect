"""Implements processors to convert examples to input and outputs, this can be
with integrarting patterns/verbalizers for PET or without."""
import abc 
import string 
from collections import OrderedDict

from fewshot.data.tasks import WiC
from .utils import Text, get_verbalization_ids, remove_final_punctuation, lowercase 


class AbstractProcessor(abc.ABC):
    def __init__(self, tokenizer, with_pattern, pattern_id=None, mask_position=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.mask_token_id
        self.mask_token = tokenizer.mask_token
        self.with_pattern = with_pattern 
        self.pattern_id = pattern_id
        self.tokenized_verbalizers = None  
        self.mask_position = mask_position

    def get_sentence_parts(self, example, mask_length):
        pass 

    def get_prompt_parts(self, example, mask_length):
         pass 

    def get_verbalizers(self):
        pass 

    def get_target(self, example):
        return example["label"]

    def get_tokenized_verbalizers(self, example=None):
       """If verbalizers are fixed per examples, this returns back a computed tokenized 
       verbalizers, but if this is example dependent, it computes the tokenized verbalizers
       per example. In this function, as a default, we compute the static one."""
       if self.tokenized_verbalizers is not None:
            return self.tokenized_verbalizers

       verbalizers = self.get_verbalizers()
       assert len(verbalizers) != 0, "for using static tokenized verbalizers computation, the length"
       "of verbalizers cannot be empty."
       self.tokenized_verbalizers=[[get_verbalization_ids(word=verbalizer, tokenizer=self.tokenizer)] for verbalizer in verbalizers]
       return self.tokenized_verbalizers

    def get_extra_fields(self, example=None):
       # If there is a need to keep extra information, here we keep a dictionary
       # from keys to their values.
       return {} 

    def get_classification_parts(self, example):
          pass 
   
    def get_parts_with_setting_masks(self, part_0, part_1, masks):
        "Only used in case of two sentences: 0`: [p,h,m],[]  `1`: [p,m,h],[]  `2`: [p],[m,h] , `3`: [p],[h,m]"
        if self.mask_position == '0':
            return part_0+part_1+masks, []
        elif self.mask_position == '1':
            return part_0+masks+part_1, []
        elif self.mask_position == '2':
            return part_0, masks+part_1
        elif self.mask_position == '3':
            return part_0, part_1+masks 



class MR(AbstractProcessor):
    name = "mr"
 
    def get_classification_parts(self, example):
        return example["source"], None 

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["source"], shortenable=True)]+mask_length*[Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        # source: https://github.com/shmsw25/Channel-LM-Prompting/blob/master/util.py
        source=Text(text=example["source"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]
        if self.pattern_id == 0:
            return [source, Text(text="A"), *masks, Text(text="one.")], []
        elif self.pattern_id == 1:
            return [source, Text(text="It was"), *masks, Text(text=".")], []
        elif self.pattern_id == 2:
            return [source, Text(text="All in all"), *masks, Text(text=".")] , []
        elif self.pattern_id == 3:
            return [source, Text(text="A"), *masks, Text(text="piece.")], []

    def get_verbalizers(self):
        return ["terrible", "great"]
    

class CR(MR):
    name = "cr"
 

class SST2(MR):
    name = "SST-2"
 

class SST5(MR):
    name = "sst-5"
 
    def get_verbalizers(self):
        return ["terrible", "bad", "okay", "good", "great"]


class Subj(AbstractProcessor):
    name = "subj"
 
    def get_classification_parts(self, example):
        return example["source"], None 

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["source"], shortenable=True)]+mask_length*[Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        # source: https://github.com/shmsw25/Channel-LM-Prompting/blob/master/util.py
        source=Text(text=example["source"], shortenable=True)
        masks= mask_length*[Text(text=self.mask_token)]
        if self.pattern_id == 0:
            return [source, Text(text="This is"), *masks, Text(text=".")], []
        elif self.pattern_id == 1:
            return [source, Text(text="It's all"), *masks, Text(text=".")], []
        elif self.pattern_id == 2:
            return [source, Text(text="It's"), *masks, Text(text=".")], []
        elif self.pattern_id == 3:
            return [source, Text(text="Is it"), *masks, Text(text="?")], []

    def get_verbalizers(self):
        return ["subjective", "objective"]
     

class Trec(AbstractProcessor):
    name = "trec"
 
    def get_classification_parts(self, example):
        return example["source"], None 

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["source"], shortenable=True)]+mask_length*[Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        # source: https://github.com/shmsw25/Channel-LM-Prompting/blob/master/util.py
        source = Text(text=example["source"], shortenable=True)
        masks=mask_length*[Text(text=self.mask_token)]
        if self.pattern_id == 0:
            return [source, *masks, Text(text=":")], []
        elif self.pattern_id == 1:
            return [source, Text(text="Q:"), *masks, Text(text=":")], []
        elif self.pattern_id == 2:
            return [source, Text(text="why"), *masks, Text(text="?")] , []
        elif self.pattern_id == 3:
            return [source, Text(text="Answer:"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return  ["Description", "Entity", "Expression", "Human", "Location", "Number"]


class BoolQ(AbstractProcessor):
    name = "boolq"

    def get_classification_parts(self, example):
        return example["passage"], example["question"] 

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return  [Text(text=example["passage"], shortenable=True)]+\
                    [Text(text=example["question"], shortenable=True)]+\
                    mask_length*[Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        passage = Text(text=example["passage"], shortenable=True)
        question = Text(text=example["question"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]
        if self.pattern_id < 2:
            return [passage, Text(text='. Question: '), question, 
                   Text(text='? Answer: '), *masks, Text(text='.')], []
        elif self.pattern_id < 4:
            return [passage, Text(text='. Based on the previous passage, '), 
                    question, Text(text='?'), *masks, Text(text='.')], []
        else:
            return [Text(text='Based on the following passage, '), question,
                    Text(text='?'), *masks, Text(text='.'), passage], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["No", "Yes"]
        return ["false", "true"]


class RTE(AbstractProcessor):
    name = "rte"
    
    def get_classification_parts(self, example):
        return example["premise"], example["hypothesis"] 

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            masks = mask_length*[Text(text=self.mask_token)]
            # TODO: we can also remove punctuation, change order of premise and hypothesis with changing the mask position.
            #return [Text(text=example["hypothesis"], shortenable=True)], [*masks, Text(text=example["premise"], shortenable=True)]
            if self.mask_position is None: 
                return [Text(text=example["premise"], shortenable=True), Text(text=example["hypothesis"], shortenable=True), *masks], []
            else:
                return self.get_parts_with_setting_masks(part_0=[Text(text=example["premise"], shortenable=True)], 
                                                         part_1=[Text(text=example["hypothesis"], shortenable=True)],
                                                         masks=masks)

        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        premise = Text(text=example["premise"], shortenable=True)
        hypothesis = Text(text=example["hypothesis"].rstrip(string.punctuation), shortenable=True)
        hypothesis_with_punctuation = Text(text=example["hypothesis"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]

        if self.pattern_id == 0:
            return [Text(text='"'), hypothesis, Text(text='" ?')], [*masks, Text(text=', "'), premise, Text(text='"')]
        elif self.pattern_id == 1:
            return [hypothesis, Text(text='?')], [*masks, Text(text=','), premise]
        if self.pattern_id == 2:
            return [Text(text='"'), hypothesis, Text(text='" ?')], [*masks, Text(text='. "'), premise, Text(text='"')]
        elif self.pattern_id == 3:
            return [hypothesis, Text(text='?')], [*masks, Text(text='.'), premise]
        elif self.pattern_id == 4:
            return [premise, Text(text=' question: '),  hypothesis_with_punctuation,
                    Text(text=' True or False? answer:'), *masks], []

    def get_verbalizers(self):
        # label order: [entialment, not-entailment] 
        if self.pattern_id == 4:
            return ['true', 'false']
        return ['Yes', 'No']


class CB(RTE):
    name = "cb"
    
    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            masks = mask_length*[Text(text=self.mask_token)]
            # TODO: we can also remove punctuation, change order of premise and hypothesis with changing the mask position.
            if self.mask_position is None:
                return [Text(text=example["premise"], shortenable=True), 
                    Text(text=example["hypothesis"], shortenable=True), 
                    *masks], []
            else:
                return self.get_parts_with_setting_masks(part_0=[Text(text=example["premise"], shortenable=True)], 
                                                         part_1=[Text(text=example["hypothesis"], shortenable=True)],
                                                         masks=masks)
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        if self.pattern_id == 4: 
            premise = Text(text=example["premise"], shortenable=True)
            hypothesis = Text(text=example["hypothesis"], shortenable=True)
            masks = mask_length*[Text(text=self.mask_token)]
            return [premise, Text(text=' question: '), hypothesis, Text(text=' true, false or neither? answer:'), *masks], []
        return super().get_prompt_parts(example, mask_length=mask_length) 
            
    def get_verbalizers(self):
        # label order: entailment, contradiction, neutral 
        if self.pattern_id == 4:
            return ['true', 'false', 'neither']
        return ['Yes', 'No', 'Maybe']


class WiC(AbstractProcessor):
    name = "wic"

    def get_classification_parts(self, example):
        return example["word"]+": "+example["sentence1"], example["sentence2"]

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            masks = mask_length*[Text(text=self.mask_token)]
            if self.mask_position is None: 
                return [Text(text=example["sentence1"], shortenable=True), 
                    Text(text=example["sentence2"], shortenable=True),
                    Text(text=example["word"]), 
                    *masks], []
            else:
                return self.get_parts_with_setting_masks(part_0=[Text(text=example["word"]), Text(text=example["sentence1"], shortenable=True)], 
                                                         part_1=[Text(text=example["sentence2"], shortenable=True)],
                                                         masks=masks)
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence1 = Text(text=example["sentence1"], shortenable=True)
        sentence2 = Text(text=example["sentence2"], shortenable=True)
        word = example["word"]
        masks = mask_length*[Text(text=self.mask_token)]

        if self.pattern_id == 0:
            return [Text(text='"'), sentence1, Text(text='" / "'), sentence2,
                    Text(text='" Similar sense of "'+word+'"?'), *masks, Text(text='.')], []
        if self.pattern_id == 1:
            return [sentence1, sentence2, Text(text='Does ' + word + ' have the same meaning in both sentences?'), *masks], []
        if self.pattern_id == 2:
            return [Text(text=example["word"]), Text(text=' . Sense (1) (a) "'), sentence1,
                    Text(text='" ('), *masks, Text(text=') "'), sentence2, Text(text='"')], []
        
    def get_verbalizers(self):
        if self.pattern_id == 2:
            return ["2", "b"]
        return ["No", "Yes"]
        

class QNLI(AbstractProcessor):
    name = "qnli"

    def get_classification_parts(self, example):
        return example["sentence"], example["question"]

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            masks = mask_length*[Text(text=self.mask_token)]
            if self.mask_position is None: 
                return [Text(text=example["question"], shortenable=True)]+\
                 [Text(text=example["sentence"], shortenable=True)]+masks, []
            else:
                return self.get_parts_with_setting_masks(part_0=[Text(text=example["sentence"], shortenable=True)],
                                                         part_1=[Text(text=example["question"], shortenable=True)], 
                                                         masks=masks)
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["sentence"], shortenable=True)
        question = Text(text=example["question"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]
        if self.pattern_id < 2:
            return [sentence, Text(text='. Question: '), question, 
                   Text(text='? Answer: '), *masks, Text(text='.')], []
        elif self.pattern_id < 4:
            return [sentence, Text(text='. Based on the previous sentence, '), 
                    question, Text(text='?'), *masks, Text(text='.')], []
        else:
            return [Text(text='Based on the following sentence, '), question,
                    Text(text='?'), *masks, Text(text='.'), sentence], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["Yes", "No"]
        return ["true", "false"]



class QQP(AbstractProcessor):
    name = "qqp"

    def get_classification_parts(self, example):
        return example["question1"], example["question2"]

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            masks = mask_length*[Text(text=self.mask_token)]
            if self.mask_position is None: 
                return [Text(text=example["question1"], shortenable=True)]+\
                    [Text(text=example["question2"], shortenable=True)]+masks, []
            else:
                return self.get_parts_with_setting_masks(part_0=[Text(text=example["question1"], shortenable=True)], 
                                                         part_1=[Text(text=example["question2"], shortenable=True)],
                                                         masks=masks)
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        question1 = Text(text=example["question1"], shortenable=True)
        question2 = Text(text=example["question2"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]
        if self.pattern_id < 2:
            return [Text(text='Do '), question1, Text(text=' and '), question2, 
                   Text(text=' have the same meaning? '), *masks, Text(text='.')], []
        elif self.pattern_id < 4:
            return [question1, Text(text='. Based on the previous question, '), 
                    question2, Text(text='?'), *masks, Text(text='.')], []
        else:
            return [Text(text='Based on the following question, '), question1,
                    Text(text='?'), *masks, Text(text='.'), question2], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["No", "Yes"]
        return ["false", "true"]

class MRPC(AbstractProcessor):
    name = "mrpc"

    def get_classification_parts(self, example):
        return example["sentence1"], example["sentence2"]

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            masks = mask_length*[Text(text=self.mask_token)]
            if self.mask_position is None: 
                return [Text(text=example["sentence1"], shortenable=True)]+\
                    [Text(text=example["sentence2"], shortenable=True)]+masks, []
            else:
                return self.get_parts_with_setting_masks(part_0=[Text(text=example["sentence1"], shortenable=True)], 
                                                         part_1=[Text(text=example["sentence2"], shortenable=True)],
                                                         masks=masks)
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence1 = Text(text=example["sentence1"], shortenable=True)
        sentence2 = Text(text=example["sentence2"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]
        if self.pattern_id < 2:
            return [Text(text='Do '), sentence1, Text(text=' and '), sentence2, 
                   Text(text=' have the same meaning? '), *masks, Text(text='.')], []
        elif self.pattern_id < 4:
            return [sentence1, Text(text='. Based on the previous sentence, '), 
                    sentence2, Text(text='?'), *masks, Text(text='.')], []
        else:
            return [Text(text='Based on the following sentence, '), sentence1,
                    Text(text='?'), *masks, Text(text='.'), sentence2], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["No", "Yes"]
        return ["false", "true"]


PROCESSOR_MAPPING = OrderedDict(
    [
        ('mr', MR),
        ('cr', CR),
        ('subj', Subj),
        ('trec', Trec),
        ('SST-2', SST2),
        ('sst-5', SST5),
        #superglue datasets 
        ('boolq', BoolQ),
        ('rte', RTE),
        ('cb', CB),
        ('wic', WiC),
        #glue datasets 
        ('qnli', QNLI),
        ('qqp', QQP),
        ('mrpc', MRPC)
    ]
)

class AutoProcessor:
    @classmethod
    def get(self, task, tokenizer, with_pattern, pattern_id, mask_position):
        if task in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[task](
                tokenizer=tokenizer,
                with_pattern=with_pattern,
                pattern_id=pattern_id,
                mask_position=mask_position)
        raise ValueError(
            "Unrecognized task {} for AutoProcessor: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in PROCESSOR_MAPPING.keys())
            )
        )

