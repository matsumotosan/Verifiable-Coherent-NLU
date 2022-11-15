import os, sys
import json
import torch
import random
import spacy
import torch

from argparse import ArgumentParser
from src.utils import get_components


def main(args):
    # Get model-related components (LM and tokenizer)
    model_class, config_class, emb_class, tokenizer, lm_class = get_components(args.model, args.cache_dir)

    # Preprocess data


    # Set up for training
    
    
    # Train model
    

    # Test model



if __name__ == "__main__":
    parser = ArgumentParser(description="Train language model on TRIP with different objectives.")

    # Model
    parser.add_argument("--dataset", type=str, default="trip")
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--objective", type=str, default="default")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    
    args = parser.parse_args()
    
    main(args)