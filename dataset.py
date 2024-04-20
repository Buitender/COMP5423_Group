import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import get_tokenizer

IGNORE_INDEX = -100

class MyDataset(Dataset):
    def __init__(self, data_path, data_partition, model_name, max_seq_len, lm_labels=True):
        self.data_path = data_path
        self.data_partition = data_partition
        self.tokenizer = get_tokenizer(model_name)
        self.max_seq_len = max_seq_len
        self.pad = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.bos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.eos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self._cache_instances()
        self.lm_labels = lm_labels

    def _cache_instances(self):
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                line_data = json.loads(line)
                conversation = line_data['conversation']
                history = []
                start = 0


                if conversation[0].startswith("bot"):
                    start = 1
                    history.append(conversation[0].strip())
                
                for i in range(start, len(conversation)-1, 2):
                    if conversation[i].startswith("bot"):
                        i += 1
                    
                    input_utterance = "history[" + " ".join(history) + "] "+" question: " + conversation[i].strip().split(": ", 1)[1]
                    
                    bot_utterance = conversation[i + 1].strip().split(": ", 1)[1]

                    history.append(conversation[i].strip())
                    history.append(conversation[i + 1].strip())
                    
                    input_utterance = self.tokenizer.encode(input_utterance)
                    input_utterance = input_utterance[-500:]
                    
                    bot_utterance = self.tokenizer.encode(bot_utterance)
                    
                    self.data.append((input_utterance, bot_utterance))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index][0]
            response = self.data[index][1]
        else:
            history = self.data[index][0]
            response = []
        return self._process(history, response)

    def _process(self, history, response):
        # truncate previous tokens if dialogue context is too long
        if len(history) > self.max_seq_len - 1:
            input_ids = [self.bos] + history[-self.max_seq_len+1:]
        else:
            input_ids = [self.bos] + history
        decoder_input_ids = [self.bos] + response
        target_ids = response + [self.eos]

        instance = {}
        instance["input_ids"] = input_ids
        instance["decoder_input_ids"] = decoder_input_ids
        instance["labels"] = target_ids
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch], batch_first=True, padding_value=self.pad)
        decoder_input_ids = pad_sequence(
            [torch.tensor(instance["decoder_input_ids"], dtype=torch.long) for instance in batch], batch_first=True, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["labels"], dtype=torch.long) for instance in batch], batch_first=True, padding_value=IGNORE_INDEX)

        return input_ids, decoder_input_ids, labels