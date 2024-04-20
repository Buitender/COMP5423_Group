import os
import torch
import random
import numpy as np

import nltk
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from transformers import BartTokenizer, BertTokenizer, GPT2Tokenizer

nltk.download('punkt')

def calculate_bleu_2(reference, candidate):
    try:
        bleu_1 = bleu_score.sentence_bleu(
                [reference], candidate,
                smoothing_function=SmoothingFunction().method1,
                weights=[1, 0, 0, 0])
    except:
        bleu_1 = 0
        try:
            bleu_2 = bleu_score.sentence_bleu(
                [reference], candidate,
                smoothing_function=SmoothingFunction().method1,
                weights=[0.5, 0.5, 0, 0])
        except:
            bleu_2 = 0

    return (bleu_1+bleu_2)/2

def calc_bleu(hyps, refs):
    """ Calculate bleu 1/2 """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return (bleu_1+bleu_2)/2

def calculate_rouge_l(reference, candidate):
    if not candidate:
        return 0
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]['rouge-l']["f"]

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_accelerator():
    if 'COLAB_TPU_ADDR' in os.environ:
        return 'xla:tpu'  # Selects TPU if available, typically in Google Colab.
    elif torch.cuda.is_available():
        return 'cuda'  # Selects GPU if available.
    elif torch.backends.mps.is_available():
        return 'mps'  # Selects Apple's Metal Performance Shaders if available.
    else:
        return 'cpu'

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

def save_model(model, optimizer, epoch, step, file_path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path+"bart_chin.pth")

def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']

    return model, optimizer, epoch, step

if __name__ == "__main__":
    references = ["I love you.", "I miss you."]
    candidates = ["I love you too.", "I miss you too."]

    
    # print("BLEU-2 Scores:", bleu_scores)
    # print("ROUGE-L Scores:", rouge_scores)