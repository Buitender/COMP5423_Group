import time
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from transformers import BartForConditionalGeneration, GPT2Tokenizer, BartTokenizer, BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import json
from nltk.translate.bleu_score import corpus_bleu

IGNORE_INDEX = -100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tokenizer(name="bert"):
    if name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif name == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    elif name == "bart_chinese":
        tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    else:
        raise ValueError("Invalid tokenizer name")
    return tokenizer


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
                    history.append(conversation[0].replace("bot:", "").strip())
                for i in range(start, len(conversation)-1, 2):
                    if conversation[i].startswith("bot"):
                        i += 1
                    user_utterance = conversation[i].replace("user:", "").strip()
                    history.append(user_utterance)
                    input_utterance = " ".join(history)
                    bot_utterance = conversation[i + 1].replace("bot:", "").strip()
                    history.append(bot_utterance)
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


class MyTrainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,
            model, train_loader, dev_loader, log_steps, num_epochs, lr, validate_steps=10, warmup_ratio=0.1, weight_decay=0.01, \
            max_grad_norm=1.0, device="cpu"
        ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_steps = log_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.validate_steps = validate_steps
        self.device = device

        if train_loader is not None:
            total_steps = len(train_loader) * self.num_epochs
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, no_deprecation_warning=True)
            self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.warmup_ratio * total_steps, num_training_steps=total_steps)
            self.best_metric = 0.0

    def train(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        print("Total batches per epoch : {}".format(len(self.train_loader)), flush=True)
        train_steps = 0
        for epoch in range(self.num_epochs):
            print("\nEpoch {}:".format(epoch + 1), flush=True)
            for batch_step, inputs in enumerate(self.train_loader):
                self.model.train()
                train_steps += 1

                input_ids, decoder_input_ids, labels = tuple(input_tensor.to(self.device) for input_tensor in inputs)
                lm_output = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_dict=True
                )
                loss = lm_output["loss"]

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if train_steps > 0 and train_steps % self.log_steps == 0:
                    print ("Train Step: {}\ttotal_loss: {:.3f}".format(train_steps, loss.item()))

                if train_steps > 0 and train_steps % self.validate_steps == 0:
                    print("Evaluating...")
                    self.evaluate(loader=self.dev_loader)
                    # Modify according to your needs !!!
            print("Epoch {} training done.".format(epoch + 1))


    def evaluate(self, loader):
        self.model.eval()
        tokenizer = get_tokenizer("bart_chinese")
        with torch.no_grad():
            total_bleu = 0
            total_samples = 0
            for inputs in tqdm(loader):
                input_ids, decoder_input_ids, labels = tuple(input_tensor.to(self.device) for input_tensor in inputs)
                lm_output = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_dict=True
                )
                logits = lm_output["logits"]
                loss = lm_output["loss"]
                # Modify according to your needs !!!
                predictions = torch.argmax(logits, dim=-1)
                predicted_sentences = []
                for pred in predictions:
                    tokens = tokenizer.convert_ids_to_tokens(pred.tolist())
                    predicted_sentences.append(self.removeSpecialTokens(tokens))

                # Convert labels to reference sentences
                references = []
                for label in labels:
                    label_tokens = tokenizer.convert_ids_to_tokens(label.tolist())
                    # label_sentence = tokenizer.convert_tokens_to_string(label_tokens)
                    references.append([self.removeSpecialTokens(label_tokens)])

                bleu = corpus_bleu(references, predicted_sentences)
                # print("bleu:", bleu)
                # Accumulate BLEU score and sample count
                total_bleu += bleu * input_ids.size(0)
                total_samples += input_ids.size(0)
            average_bleu = total_bleu / total_samples
            print("Average BLEU Score:", average_bleu)

    def removeSpecialTokens(self, tokens):
        result = []
        for token in tokens:
            if token == '[CLS]':
                continue
            if token == '[SEP]':
                break;
            result.append(token)
        return result

    def test(self, test_loader):
        self.model.eval()
        tokenizer = get_tokenizer("bart_chinese")
        with torch.no_grad():
            total_bleu = 0
            total_samples = 0
            for inputs in test_loader:
                input_ids, decoder_input_ids, labels = tuple(input_tensor.to(self.device) for input_tensor in inputs)
                lm_output = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_dict=True
                )
                logits = lm_output["logits"]
                loss = lm_output["loss"]
                # Modify according to your needs !!!
                predictions = torch.argmax(logits, dim=-1)
                predicted_sentences = []
                for pred in predictions:
                    tokens = tokenizer.convert_ids_to_tokens(pred.tolist())
                    tokens = self.removeSpecialTokens(tokens)
                    predicted_sentences.append(tokens)

                references = []
                for label in labels:
                    label_tokens = tokenizer.convert_ids_to_tokens(label.tolist())
                    # label_sentence = tokenizer.convert_tokens_to_string(label_tokens)
                    references.append([self.removeSpecialTokens(label_tokens)])

                bleu = corpus_bleu(references, predicted_sentences)
                # print("bleu:", bleu)
                # Accumulate BLEU score and sample count
                total_bleu += bleu * input_ids.size(0)
                total_samples += input_ids.size(0)
            average_bleu = total_bleu / total_samples
            print("Average BLEU Score:", average_bleu)


def main():
    seed = 42
    train_file = "data/train.txt"
    dev_file = "data/dev.txt"
    test_file = "data/test.txt"
    max_len = 512
    batch_size = 8
    epochs = 10
    learn_rate = 2e-5
    log_steps = 100
    validate_steps = 1000
    set_seed(seed)
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = get_tokenizer("bart_chinese")

    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    print("Preparing datasets...", flush=True)
    train_dataset = MyDataset(train_file, "train", "bart_chinese", max_len)
    dev_dataset = MyDataset(dev_file, "dev", "bart_chinese", max_len)
    test_dataset = MyDataset(test_file, "test", "bart_chinese", max_len)

    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              num_workers=8,
                              batch_size=batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dev_dataset,
                            collate_fn=dev_dataset.collate,
                            num_workers=8,
                            batch_size=batch_size,
                            shuffle=False)

    trainer = MyTrainer(model=model, train_loader=train_loader, dev_loader=dev_loader, log_steps=log_steps,
                        num_epochs=epochs, lr=learn_rate, validate_steps=validate_steps, device=device,
                        warmup_ratio=0.1, weight_decay=0.01, max_grad_norm=1.0)
    trainer.train()

    if test_file is not None:
        test_loader = DataLoader(test_dataset,
                                 collate_fn=test_dataset.collate,
                                 num_workers=8,
                                 batch_size=batch_size,
                                 shuffle=False)
        trainer.test(test_loader)
    end = time.time()
    print(f"Prcessing Time: {(end - start) / 60} min", flush=True)
    model.save_pretrained("./model")


if __name__ == "__main__":
    main()


