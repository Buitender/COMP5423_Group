import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import get_tokenizer, calculate_bleu_2, calculate_rouge_l

class MyTrainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self, logging,
            model, device, train_loader, dev_loader, log_steps, num_epochs, lr, validate_steps=10, warmup_ratio=0.1, weight_decay=0.01, \
            max_grad_norm=1.0):
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
        self.logging = logging

        if train_loader is not None:
            total_steps = len(train_loader) * self.num_epochs
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, no_deprecation_warning=True)
            self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.warmup_ratio * total_steps, num_training_steps=total_steps)
            self.best_metric = 0.0

    def train(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        self.logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        
        train_steps = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0

            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}', leave=False)
            
            for inputs in progress_bar:
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

                epoch_loss += loss.item()

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if train_steps > 0 and train_steps % self.log_steps == 0:
                    self.logging.info("Train Step: {}\ttotal_loss: {:.3f}".format(train_steps, loss.item()))

                if train_steps > 0 and train_steps % self.validate_steps == 0:
                    avg_bleu, avg_rouge, avg_loss = self.evaluate(loader=self.dev_loader)
                    self.logging.info(f'Evaluation Step {train_steps} - Bleu: {avg_bleu:.3f}, Rouge: {avg_rouge:.3f}, Loss: {avg_loss:.3f}')

                    if avg_bleu + avg_rouge > self.best_metric:
                        self.best_metric = avg_bleu + avg_rouge
                        self.logging.info("Saving the best model...")
                        self.model.save_pretrained(f"./model/{epoch+1}_{train_steps}/")
                        self.logging.info("Best model saved.")
                    
            self.logging.info("Epoch {} training done.".format(epoch + 1))


    def evaluate(self, loader):
        self.model.eval()
        tokenizer = get_tokenizer("bart_chinese")
        bleu_scores = []
        rouge_scores = []
        total_loss = 0

        with torch.no_grad():
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
                total_loss += loss.item()

                pred_ids = torch.argmax(logits, dim=-1)

                for pred_id, label in zip(pred_ids, labels):
                    pre_text = tokenizer.decode(pred_id, skip_special_tokens=True)
                    ref_text = tokenizer.decode(label, skip_special_tokens=True)

                    pre_token = tokenizer.tokenize(pre_text)
                    ref_token = tokenizer.tokenize(ref_text)
                    
                    bleu_score = calculate_bleu_2(pre_token, ref_token)
                    rouge_score = calculate_rouge_l(ref_text, pre_text)

                                
                    bleu_scores.append(bleu_score)
                    rouge_scores.append(rouge_score)
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_loss = total_loss / len(loader)

        return avg_bleu, avg_rouge, avg_loss
    
    def removeSpecialTokens(self, tokens):
        result = []
        for token in tokens:
            if token == '[CLS]':
                continue
            if token == '[SEP]':
                break;
            result.append(token)
        return result


def inference(model, tokenizer, device, input_text):
    model.eval()

    if isinstance(input_text, str):
        input_text = [input_text]

    results = []
    for text in input_text:
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(inputs, num_beams=5, early_stopping=True)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(generated_text)