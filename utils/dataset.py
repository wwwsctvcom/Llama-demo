import torch
import json
from torch.utils.data import Dataset


class DataBuilder:

    def __init__(self,
                 train_path: str = None,
                 test_path: str = None):
        self.lines = []
        with open(train_path, "r", encoding="utf-8") as reader:
            self.lines += reader.readlines()

        with open(test_path, "r", encoding="utf-8") as reader:
            self.lines += reader.readlines()

        self.pad = "<PAD>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.vocab = [self.pad, self.bos, self.eos]
        for line in self.lines:
            for word in json.loads(line)["instruction"]:
                if word != " ":
                    self.vocab.append(word)
            for word in json.loads(line)["input"]:
                if word != " ":
                    self.vocab.append(word)
            for word in json.loads(line)["output"]:
                if word != " ":
                    self.vocab.append(word)
        self.vocab = sorted(list(set(self.vocab)))

    def get_vocab(self):
        return self.vocab

    def get_id2word(self) -> dict:
        return {index: word for index, word in enumerate(self.vocab)}

    def get_word2id(self):
        return {word: index for index, word in self.get_id2word().items()}


class LlamaDataset(Dataset):
    def __init__(self,
                 train_path: str = None,
                 test_path: str = None,
                 word2id: dict = None,
                 max_seq_length: int = 1024,
                 is_skip: bool = True):
        self.lines = []
        with open(train_path, "r", encoding="utf-8") as reader:
            self.lines += reader.readlines()

        with open(test_path, "r", encoding="utf-8") as reader:
            self.lines += reader.readlines()
        if self.lines is None:
            raise ValueError("lines is None")

        self.word2id = word2id
        if self.word2id is None:
            raise ValueError("word2id is None")
        self.max_seq_length = max_seq_length
        self.is_skip = is_skip

        # special token
        self.pad = "<PAD>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"

        # examples all
        self.examples = []
        for line in self.lines:
            skip_flag = False
            src_tokens = json.loads(line)["instruction"] + json.loads(line)["input"]
            if len(src_tokens) > self.max_seq_length:
                skip_flag = True
            src_tokens = [word for word in src_tokens]
            src_ids = [word2id[word] for word in [self.bos] + src_tokens]

            tgt_tokens = json.loads(line)["output"]
            if len(tgt_tokens) > max_seq_length:
                skip_flag = True
            tgt_tokens = [word for word in tgt_tokens]

            tgt_ids = [word2id[word] for word in tgt_tokens + [self.eos]]

            input_ids = src_ids + tgt_ids
            labels = [-100] * len(src_ids) + tgt_ids
            attention_mask = [0 if t == word2id[self.pad] else 1 for t in input_ids]

            if skip_flag and self.is_skip:
                continue

            self.examples.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


data_builder = DataBuilder(train_path="./data/train.txt", test_path="./data/test.txt")
word2id = data_builder.get_word2id()
id2word = data_builder.get_id2word()


def data_collator(batch: list):
    lengths = [len(instance["input_ids"]) for instance in batch]
    # padding by max len
    batch_max_len = max(lengths)

    input_ids_batch, labels_batch, attention_mask_batch = [], [], []
    for instance in batch:
        input_ids = instance["input_ids"]
        labels = instance["labels"]
        attention_mask = instance["attention_mask"]
        # paddings
        padding_len = batch_max_len - len(input_ids)
        input_ids = [word2id["<PAD>"]] * padding_len + input_ids
        labels = [-100] * padding_len + labels
        attention_mask = [0] * padding_len + attention_mask
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        attention_mask_batch.append(attention_mask)
    return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long)}
