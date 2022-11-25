import os

from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DebertaForSequenceClassification
from transformers import BertModel, RobertaModel, DebertaModel
from transformers import RobertaForMaskedLM
from transformers import BertConfig, RobertaConfig, DebertaConfig


model_dict = {
    "bert": "bert-large-uncased",
    "bert_piqa": "sledz08/finetuned-bert-piqa",
    "roberta": "roberta-large",
    "roberta_mnli": "roberta-large-mnli",
    "roberta_large_squad": "deepset/robert-large-squad2",
    "roberta_large_race": "LIAMF-USP/roberta-large-finetuned-race",
    "deberta": "microsoft/deberta-base",
    "deberta_large": "microsoft/deberta-large",
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
    elif model in ['roberta', 'roberta_mnli', 'roberta_large_squad', 'roberta_large_race']:
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

    # Get tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        do_lower_case=False,
        cache_dir=cache_dir
    )

    return model_name, model_class, config_class, emb_class, tokenizer, lm_class