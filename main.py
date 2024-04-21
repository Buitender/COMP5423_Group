import os
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration

from utils import set_seed, get_tokenizer, select_accelerator
from dataset import MyDataset
from trainer import MyTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure the script for training or inference.')

    # Optional arguments with default values
    parser.add_argument('--filefolder_path', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length of the sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training or inference')
    parser.add_argument('--num_epoch', type=int, default=20, help='Number of epochs (ignored if in inference mode)')
    parser.add_argument('--learn_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warm-up ratio')
    parser.add_argument('--log_steps', type=int, default=100, help='Number of steps to log training progress')
    parser.add_argument('--validate_steps', type=int, default=1000, help='Number of steps to validate training progress')
    parser.add_argument('--use_type', type=str, choices=['train', 'inference'], default='train', help='Select the mode: train or inference')
    parser.add_argument('--load_model', action='store_true', help='Load model from a file instead of initializing')
    parser.add_argument('--model_path', type=str, default='./model/', help='Path to the model folder')

    # Parse the command line arguments
    args = parser.parse_args()

    # Set seed and select device
    set_seed(42)
    device = select_accelerator()

    # Print arguments (optional, for verification)
    print(f"Operating mode: {args.use_type}")
    print(f"Training configuration: {args}")

    if args.use_type == 'train':
        print(f"The data will be loaded from: {args.filefolder_path}")
        print(f"Training will proceed for {args.num_epoch} epochs with a batch size of {args.batch_size}.")
        logging.basicConfig(filename="train.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        tokenizer = get_tokenizer("bart_chinese")
        if args.load_model:
            try:
                model = BartForConditionalGeneration.from_pretrained(args.model_path)
            except:
                Warning("Model not found. Initializing a new model.")
                model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        else:
            model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        
        print("Preparing datasets...", flush=True)
        train_dataset = MyDataset(args.filefolder_path + "/train.txt", "train", "bart_chinese", args.max_len)
        dev_dataset = MyDataset(args.filefolder_path + "./dev.txt", "dev", "bart_chinese", args.max_len)
        

        train_loader = DataLoader(train_dataset,
                                collate_fn=train_dataset.collate,
                                num_workers=8,
                                batch_size=args.batch_size,
                                shuffle=True)
        dev_loader = DataLoader(dev_dataset,
                                collate_fn=dev_dataset.collate,
                                num_workers=8,
                                batch_size=args.batch_size,
                                shuffle=False)

        trainer = MyTrainer(model=model, logging=logging, train_loader=train_loader, dev_loader=dev_loader, log_steps=args.log_steps,
                            num_epochs=args.num_epoch, lr=args.learn_rate, validate_steps=args.validate_steps, device=device,
                            warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=1.0)
        trainer.train()
    elif args.use_type == 'inference':
        print(f"Inference mode: Only using batch size of {args.batch_size} for processing.")
        logging.basicConfig(filename="inference.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        tokenizer = get_tokenizer("bart_chinese")
        if args.load_model:
            try:
                model = BartForConditionalGeneration.from_pretrained(args.model_path)
            except:
                Warning("Model not found. Initializing a new model.")
                model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        else:
            model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        if args.filefolder_path + "/test.txt" is not None:
            test_dataset = MyDataset(args.filefolder_path + "/test.txt", "test", "bart_chinese", args.max_len)
            test_loader = DataLoader(test_dataset,
                                 collate_fn=test_dataset.collate,
                                 num_workers=8,
                                 batch_size=args.batch_size,
                                 shuffle=False)
            
            trainer = MyTrainer(model=model, logging=logging, train_loader=None, dev_loader=None, log_steps=args.log_steps,
                            num_epochs=args.num_epoch, lr=args.learn_rate, validate_steps=args.validate_steps, device=device,
                            warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=1.0)
            
            result = trainer.inference(test_loader)

            with open(args.filefolder_path+"/result.txt", "w") as f:
                for item in result:
                    f.write("%s\n" % item)
        else:
            raise ValueError("Please provide the path to the test file for inference.")