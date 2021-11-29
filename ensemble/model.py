import numpy as np

import torch
import csv
import urllib
from torch import nn
from transformers import BertPreTrainedModel, BertConfig, BertModel, AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification

class BertEnsemble(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        task = 'offensive'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        self.dialog_cls_model = BertForSequenceClassification.from_pretrained("./saved_models/bert_base_uncased_4epoch_5historycontext_sepCorrected2", num_labels=3).to('cuda')
        # model for AQ

        self.offense_model = AutoModelForSequenceClassification.from_pretrained(MODEL).to('cuda')
        self.offense_model.save_pretrained(MODEL)
        # combine the 2 models into 1
        self.cls = nn.Linear(3+len(labels), 3)



    def forward(self, input):
        out_dl = self.dialog_cls_model(**input)
        out_offense = self.offense_model(**input)

        out = torch.cat([out_dl[0],out_offense[0]], dim=1)

        logits = self.cls(out)

        return logits