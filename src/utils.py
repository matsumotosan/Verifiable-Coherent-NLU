import os

from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, GPT2Tokenizer, BartTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DebertaForSequenceClassification, GPT2ForSequenceClassification, BartForSequenceClassification
from transformers import BertModel, RobertaModel, DebertaModel, GPT2Model, BartModel
from transformers import RobertaForMaskedLM
from transformers import BertConfig, RobertaConfig, DebertaConfig, GPT2Config, BartConfig


model_dict = {
    "bert": "bert-large-uncased",
    "bert_piqa": "sledz08/finetuned-bert-piqa",
    "roberta": "roberta-large",
    "roberta_mnli": "roberta-large-mnli",
    "roberta_large_squad": "deepset/roberta-large-squad2",
    "roberta_large_xlm_squad": "deepset/xlm-roberta-large-squad2",
    "roberta_large_race": "LIAMF-USP/roberta-large-finetuned-race",
    "deberta": "microsoft/deberta-base",
    "deberta_large": "microsoft/deberta-large",
    "gpt2": "distilgpt2",
    "gpt2-large": "gpt2-large",
    "bart": "facebook/bart-base", 
}


def get_components(model, cache_dir):
    # Get name of model
    model_name = model_dict[model]

    # Get model, config, embedding, and tokenizer class
    if model in ['bert', 'bert_piqa']:
        model_class = BertForSequenceClassification
        config_class = BertConfig
        emb_class = BertModel
        tokenizer_class = BertTokenizer
        lm_class = None
    elif model in ['roberta', 'roberta_mnli', 'roberta_large_squad', 'roberta_large_xlm_squad', 'roberta_large_race']:
        model_class = RobertaForSequenceClassification
        config_class = RobertaConfig
        emb_class = RobertaModel
        lm_class = RobertaForMaskedLM
        tokenizer_class = RobertaTokenizer
    elif model in ['deberta', 'deberta_large']:
        model_class = DebertaForSequenceClassification
        config_class = DebertaConfig
        emb_class = DebertaModel
        tokenizer_class = DebertaTokenizer
        lm_class = None
    elif model in ['gpt2', 'gpt2-large']:
        model_class = GPT2ForSequenceClassification
        config_class = GPT2Config
        emb_class = GPT2Model
        tokenizer_class = GPT2Tokenizer
        lm_class = None
    elif model == 'bart':
        model_class = BartForSequenceClassification
        config_class = BartConfig
        emb_class = BartModel
        tokenizer_class = BartTokenizer
        lm_class = None

    # Get tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        do_lower_case=False,
        cache_dir=cache_dir
    )

    return model_name, model_class, config_class, emb_class, tokenizer, lm_class