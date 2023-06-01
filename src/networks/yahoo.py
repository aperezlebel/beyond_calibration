import re

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset, Subset
from transformers import BartForSequenceClassification, BartTokenizer

from .base import BaseNet


class ToPyTorchDataset(Dataset):
    """Convert HuggingFace dataset to PyTorch dataset."""

    def __init__(self, dset, keys=None):
        self.dset = dset
        self.keys = keys

    def __getitem__(self, idx):
        d = self.dset[idx]
        if self.keys is None:
            return d
        return [d[k] for k in self.keys]

    def __len__(self):
        return len(self.dset)


class ZeroShotBartYahoo(BaseNet):
    def __init__(self, split="test_unseen"):
        self.split = split
        self.hg_model_name = "joeddav/bart-large-mnli-yahoo-answers"
        super().__init__()

    def get_tokenizer(self):
        return BartTokenizer.from_pretrained(self.hg_model_name)

    def create_model(self):
        return BartForSequenceClassification.from_pretrained(self.hg_model_name)

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classification_head.out_proj
        model.classification_head.out_proj = torch.nn.Identity()
        return model, last_layer

    def forward_truncated(self, input):
        return self.truncated_model.forward(torch.tensor(input))[0]

    def forward_whole(self, input):
        return self.model.forward(torch.tensor(input))[0]

    def get_dataset(self):
        dataset = load_dataset("yahoo_answers_topics")

        r = re.match(r"(train|test)_(seen|unseen|all)(_(.*))?", self.split)

        mainsplit = r.group(1)
        subsplit = r.group(2)
        sublabel = r.group(4)

        if mainsplit in ["train", "test"]:
            ds = dataset[mainsplit]
        else:
            raise ValueError(f"Unknown split {mainsplit} for {self}.")

        d = {
            "society": "Society & Culture",
            "science": "Science & Mathematics",
            "health": "Health",
            "education": "Education & Reference",
            "computers": "Computers & Internet",
            "sports": "Sports",
            "business": "Business & Finance",
            "entertainment": "Entertainment & Music",
            "family": "Family & Relationships",
            "politics": "Politics & Government",
        }

        if subsplit == "seen":
            split_labels = self.get_seen_label_names()

        elif subsplit == "unseen":
            split_labels = self.get_unseen_label_names()

        elif subsplit == "all":
            split_labels = self.get_label_names()

        else:
            raise ValueError(f"Unknown subsplit {subsplit} for {self}.")

        if sublabel is None:
            selected_labels = split_labels

        elif sublabel in d and d[sublabel] in split_labels:
            selected_labels = [d[sublabel]]

        else:
            raise ValueError(
                f"Unknown sublabel {sublabel} for subsplit " f"{subsplit} in {self}."
            )

        all_labels = self.get_label_names()
        selected_label_ids = [all_labels.index(x) for x in selected_labels]
        split_label_ids = [all_labels.index(x) for x in split_labels]

        # Encode selected dataset with selected labels
        tokenizer = self.get_tokenizer()

        def encode(e, label_id):
            label_name = all_labels[label_id]
            hypothesis = f"This text is about {label_name}."
            token = tokenizer.encode(
                e["question_title"],
                hypothesis,
                return_tensors="pt",
                max_length=1024,
                truncation_strategy="only_first",
            )

            topic_id = e["topic"]
            label_binarized = topic_id == label_id

            return {
                "question_title_token": token,
                "hypothesis": hypothesis,
                "label_name": label_name,
                "label_id": label_id,
                "label_binarized": label_binarized,
            }

        L_ds = []
        for label_id in selected_label_ids:
            L_ds.append(ds.map(lambda e: encode(e, label_id), batched=False))

        ds_encoded = concatenate_datasets(L_ds)
        topic = ds_encoded["topic"]
        ds_encoded = ToPyTorchDataset(
            ds_encoded, keys=["question_title_token", "label_binarized"]
        )

        # Apply the subsplit selection
        idx = np.isin(topic, split_label_ids)
        idx = np.where(idx)[0]
        ds_encoded = Subset(ds_encoded, indices=idx.tolist())

        return ds_encoded

    def get_dataset_name(self):
        return f"YahooAnswersTopics_{self.split}"

    def get_w(self):
        return self.last_layer.weight.detach()

    def get_intercept(self):
        return self.last_layer.bias.detach()

    def logits_to_scores(self, y_logits):
        entail_contradiction_logits = y_logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        return probs

    def get_class_names(self):
        return ["False", "True"]

    def get_label_names(self):
        return [
            "Society & Culture",
            "Science & Mathematics",
            "Health",
            "Education & Reference",
            "Computers & Internet",
            "Sports",
            "Business & Finance",
            "Entertainment & Music",
            "Family & Relationships",
            "Politics & Government",
        ]

    def get_unseen_label_names(self):
        return [
            "Education & Reference",
            "Science & Mathematics",
            "Sports",
            "Politics & Government",
            "Entertainment & Music",
        ]

    def get_seen_label_names(self):
        return [
            "Society & Culture",
            "Health",
            "Computers & Internet",
            "Business & Finance",
            "Family & Relationships",
        ]
