import json
from collections import Counter

import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABELS = ["sports", "technology", "shopping"]

KEYWORD_MAP = {
    "sports": [
        "football", "soccer", "basketball", "nba", "nfl", "cricket", "tennis",
        "golf", "baseball", "mlb", "boxing", "surfing", "olympics", "champion",
        "league", "match", "game", "score", "race", "f1", "draft", "tournament",
        "premier", "world cup", "esports", "fitness", "gym", "running", "yoga",
    ],
    "technology": [
        "python", "javascript", "react", "vue", "kubernetes", "docker", "cloud",
        "machine learning", "ai", "gpu", "rtx", "raspberry", "cybersecurity",
        "vpn", "data engineering", "iphone", "laptop", "monitor", "framework",
        "certification", "software",
    ],
    "shopping": [
        "buy", "deal", "sale", "cheap", "discount", "price", "shop", "coupon",
        "black friday", "prime day", "clearance", "bundle",
    ],
}

stats = {
    "total_queries": 0,
    "labeled": 0,
    "unlabeled": 0,
}


def classify_query(query):
    query_lower = query.lower()
    scores = {}
    for label, keywords in KEYWORD_MAP.items():
        scores[label] = sum(1 for kw in keywords if kw in query_lower)

    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return None
    return best_label


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def process(input_path):
    with open(input_path) as f:
        data = json.load(f)

    queries = []
    labels = []

    for item in data:
        query = item["query"]
        label = classify_query(query)
        stats["total_queries"] += 1

        if label is None:
            stats["unlabeled"] += 1
            continue

        queries.append(query)
        labels.append(LABELS.index(label))
        stats["labeled"] += 1

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(LABELS)
    )

    encodings = tokenizer(queries, truncation=True, padding=True, max_length=64)
    dataset = QueryDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    predictions = trainer.predict(dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)
    acc = accuracy_score(labels, predicted_labels)

    label_dist = Counter(LABELS[l] for l in labels)
    stats["accuracy"] = round(acc, 4)
    stats["label_distribution"] = dict(label_dist)

    return stats
