"""
IMDB Sentiment Analysis: NN PyTorch
- Questions: Train models with different modifications (1 to 5)

Observation and EVAL
- Plot training metrics (loss & accuracy)
- Side-by-side review predictions
- Bar plots to show prediction probabilities
- Save and load trained models for reuse
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from torchtext.datasets import IMDB 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Specifies the device (GPU/CPU) for training. 

MAX_LEN = 200       # maximum length for each review sequence. 
NUM_CLASSES = 2  # 3   # 2 logit (classification: positive, negative, neutral)

BATCH_SIZE = 64     # number of samples in each training batch. 
EMBED_DIM = 128     # size of the vector for each word embedding
EPOCHS = 5 #10        # ttoal number of times the training process will iterate over the entire dataset
LEARN_RATE = 1e-3       # Sets the learning rate for the optimizer. 0.001 accepted

OUTPUT_DIR = "plots" # output plots
MODEL_DIR = "models" # trained models
TOKENIZER = "basic_english"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def build_imdb_loaders(max_len=MAX_LEN, batch_size=BATCH_SIZE):
    """Builds and returns the IMDB data loaders and vocabulary."""
    try:
        tokenizer = get_tokenizer(TOKENIZER)

        def yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        # vocabulary from the training set
        train_iter_for_vocab = IMDB(split="train")
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter_for_vocab),
            specials=["<unk>", "<pad>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        PADDING_INDEX = vocab["<pad>"]

        def text_pipeline(x):
            return vocab(tokenizer(x))[:max_len]

        def label_pipeline(y):
            if isinstance(y, str):
                return 1 if y == "pos" else 0
            elif isinstance(y, int):
                return 1 if y == 2 else 0
            else:
                raise ValueError(f"Unexpected label: {y}")

        def collate_batch(batch):
            batch_labels, batch_sequences = [], []
            
            for label, text in batch:
                batch_labels.append(label_pipeline(label))
                token_ids = torch.tensor(text_pipeline(text), dtype=torch.long)
                batch_sequences.append(token_ids)
            
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            batch_sequences = pad_sequence(batch_sequences, batch_first=True, padding_value=PADDING_INDEX)
            return batch_sequences, batch_labels

        train_iter, test_iter = IMDB(split=("train", "test"))
        train_data = list(train_iter)
        val_data = list(test_iter)

        print("IMDB training data")
        sample_indices = random.sample(range(len(train_data)), 10)
        for id, i in enumerate(sample_indices):
            label, text = train_data[i]
            print(f"Sample {id+1} ({label}) -- {text[:(max_len-100)]}")  # first 100 characters

        # training
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch) # random True

        # validation data
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        
        return train_loader, val_loader, vocab, NUM_CLASSES
    except Exception as e:
        print(f"Error building data loaders: {e}")
        return None, None, None, None

class SentimentFFN(nn.Module):
    """A simple FeedForward Neural Network for sentiment analysis."""

    def __init__(self, vocab_size, embed_dim, num_classes,
                 activation="relu", batch_norm=False, hidden_layers=1, hidden_dims=(128,64)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        print(f"Initializing model...\n" 
              f"Hidden Layers: {hidden_layers}, Activation: {activation}, " 
              f"Batch Norm: {batch_norm}, Vocab Size: {vocab_size}")

        if activation == "relu": activation_function = nn.ReLU() #Question 1
        elif activation == "relu6": activation_function = nn.ReLU6() # #Question 3
        elif activation == "leakyrelu": activation_function = nn.LeakyReLU(0.01) # Question 3 relu6 or leakyrelu (neg slope estimatd)
        else: raise ValueError("Invalid Activation Function specified!")

        layers = []
        input_dimension = embed_dim

        if hidden_layers == 1:
            layers.append(nn.Linear(input_dimension, hidden_dims[0]))
            if batch_norm: layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(activation_function)
            layers.append(nn.Linear(hidden_dims[0], num_classes))
        else:
            hidden_dim1, hidden_dim2 = hidden_dims[0], hidden_dims[1]
            layers.append(nn.Linear(input_dimension, hidden_dim1))
            if batch_norm: layers.append(nn.BatchNorm1d(hidden_dim1))
            layers.append(activation_function)
            layers.append(nn.Linear(hidden_dim1, hidden_dim2))
            if batch_norm: layers.append(nn.BatchNorm1d(hidden_dim2))
            layers.append(activation_function)
            layers.append(nn.Linear(hidden_dim2, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embeddings = self.embedding(x)
        # pooled_embeddings = embeddings.mean(dim=1)
        pooled_embeddings, _ = torch.max(embeddings, dim=1)
        return self.mlp(pooled_embeddings)

def train_model(name, model, train_loader, val_loader, epochs=EPOCHS, lr=LEARN_RATE, weight_decay=0.0):
    """Trains the model and returns metrics and the trained model."""
    
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n--- Training Model: {name} | Device: {DEVICE} ---")
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss() # mean
    # optimizer = optim.Adam(model.parameters(), lr, weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=weight_decay)

    # loop epochs
    for epoch in range(1, epochs + 1):
        try:
            model.train()
            total, correct, total_loss = 0, 0, 0

            for features, labels in train_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * features.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            train_loss = total_loss / total

            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(train_acc)

            model.eval()
            with torch.no_grad():
                val_total, val_correct, val_loss = 0, 0, 0
                
                for features, labels in val_loader:
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    logits = model(features)
                    
                    val_loss += nn.CrossEntropyLoss()(logits, labels).item() * features.size(0)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

                val_acc = 100 * val_correct / val_total
                val_loss = val_loss / val_total
                metrics["val_loss"].append(val_loss)
                metrics["val_acc"].append(val_acc)

            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | " 
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        except Exception as e:
            print(f"An error occurred during training epoch {epoch}: {e}")
            break
            
    return metrics, model

def plot_metrics(all_metrics, config_name):
    """Plots and saves training and validation loss and accuracy."""
    plt.figure(figsize=(14, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for name, metrics in all_metrics.items():
        plt.plot(metrics["train_loss"], label=f"{name} Train")
        plt.plot(metrics["val_loss"], '--', label=f"{name} Val")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for name, metrics in all_metrics.items():
        plt.plot(metrics["train_acc"], label=f"{name} Train")
        plt.plot(metrics["val_acc"], '--', label=f"{name} Val")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    
    try:
        filename = os.path.join(OUTPUT_DIR, f"{config_name}_metrics.png")
        plt.savefig(filename)
        print(f"Saved metrics plot to {filename}")
    except Exception as e:
        print(f"Error saving metrics plot: {e}")
    plt.close()

def plot_review_probabilities(reviews, vocab, trained_models, config_name):
    """Ploting and saving bar charts of positive sentiment probabilities for sample reviews used."""
    tokenizer_fn = get_tokenizer(TOKENIZER)#.tokenizer()
    
    input_tensors = []
    for review in reviews: # sample reviews
        token_ids = vocab(tokenizer_fn(review))[:MAX_LEN]
        tensor = torch.tensor(token_ids, dtype=torch.long)
        if len(token_ids) < MAX_LEN:
            tensor = torch.cat([tensor, torch.zeros(MAX_LEN - len(token_ids), dtype=torch.long)])
        input_tensors.append(tensor)
    
    inputs = torch.stack(input_tensors).to(DEVICE)

    for i, review in enumerate(reviews):
        probs_per_step = []
        step_names = list(trained_models.keys())
        for step_name in step_names:
            model = trained_models[step_name]
            model.eval()
            with torch.no_grad():
                logits = model(inputs[i].unsqueeze(0))
                probs = F.softmax(logits, dim=1)
                probs_per_step.append(probs.cpu().numpy().flatten())
        
        probs_per_step = np.array(probs_per_step)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(probs_per_step)), probs_per_step[:, 1], tick_label=step_names)
        plt.title(f"Review '{review[:30]}...' Positive Probability")
        plt.ylabel("Probability")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        try:
            filename = os.path.join(OUTPUT_DIR, f"{config_name}_review_{i+1}_probs.png")
            plt.savefig(filename)
            print(f"Saved review probability plot to {filename}")
        except Exception as e:
            print(f"Error saving review probability plot: {e}")
        plt.close()

def print_summary_table(all_metrics):
    """Prints a summary table of the final metrics."""
    summary_data = []
    for name, metrics in all_metrics.items():
        summary_data.append({
            "Step": name,
            "Final Train Accuracy (%)": metrics["train_acc"][-1],
            "Final Val Accuracy (%)": metrics["val_acc"][-1],
            "Final Train Loss": metrics["train_loss"][-1],
            "Final Val Loss": metrics["val_loss"][-1]
        })
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Table")
    print(summary_df)

def review_comparison_table(reviews, vocab, trained_models):
    """Creates and prints a side-by-side comparison of model predictions."""
    tokenizer_fn = get_tokenizer(TOKENIZER)
    input_tensors = []
    for review in reviews:
        token_ids = vocab(tokenizer_fn(review))[:MAX_LEN]
        tensor = torch.tensor(token_ids, dtype=torch.long)
        if len(token_ids) < MAX_LEN:
            tensor = torch.cat([tensor, torch.zeros(MAX_LEN - len(token_ids), dtype=torch.long)])
        input_tensors.append(tensor)
    inputs = torch.stack(input_tensors).to(DEVICE)

    comparison_data = {"Review": reviews}
    for step_name, model in trained_models.items():
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1) # for probability classification model
            preds = probs.argmax(dim=1)

        pred_list = ["Positive" if p == 1 else "Negative" for p in preds]
        prob_list = [f"[{p[0]:.2f}, {p[1]:.2f}]" for p in probs]
        comparison_data[f"{step_name} Pred"] = pred_list
        comparison_data[f"{step_name} Prob"] = prob_list

    comparison_df = pd.DataFrame(comparison_data)
    print("\nReview Predictions")
    print(comparison_df)
    return comparison_df

def save_models(trained_models):
    """Saves the state dictionary of trained models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for step_name, model in trained_models.items():
        try:
            path = os.path.join(MODEL_DIR, f"{step_name.replace(' ', '_')}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved {step_name} to {path}")
        except Exception as e:
            print(f"Error saving model {step_name}: {e}")

def load_models(steps_config, vocab, num_classes):
    """Loads trained models from saved state dictionaries."""
    loaded_models = {}
    print("\n--- Loading Models ---")
    for config in steps_config:
        step_name = config["name"]
        try:
            model = SentimentFFN(len(vocab), EMBED_DIM, num_classes,
                                 activation=config["activation"],
                                 batch_norm=config["batch_norm"],
                                 hidden_layers=config["hidden_layers"],
                                 hidden_dims=config["hidden_dims"])
            path = os.path.join(MODEL_DIR, f"{step_name.replace(' ', '_')}.pt")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                loaded_models[step_name] = model
                print(f"Loaded {step_name} from {path}")
            else:
                print(f"Model file not found for {step_name} at {path}")
        except Exception as e:
            print(f"Error loading model {step_name}: {e}")
    return loaded_models

def main():
    """Main function to run the sentiment analysis pipeline."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    train_loader, val_loader, vocab, num_classes = build_imdb_loaders()
    # print(f"train_loader {train_loader}")
    # print(f"val_loader {val_loader}")
    # print(f"vocab {vocab}")
    # print(f"num_classes {num_classes}")

    if not all([train_loader, val_loader, vocab, num_classes]):
        print("Failed to load data. Exiting.")
        return

    # Parameters for Question specific
    steps_config = [
            {"question":"1. Design a feedforward neural network for multi-class classification using PyTorch.", "name": "BasicFeedforward", "activation": "relu", "batch_norm": False, "hidden_layers": 1, "weight_decay": 0.0, "hidden_dims": (128,)},
            {"question":"2. Add batch normalization to a feedforward neural network in PyTorch.", "name": "BatchNorm", "activation": "relu", "batch_norm": True, "hidden_layers": 1, "weight_decay": 0.0, "hidden_dims": (128,)},
            {"question":"3. Modify the network to use ReLU6 or LeakyReLU instead of ReLU.", "name": "LeakyReLU_Activation", "activation": "leakyrelu", "batch_norm": True, "hidden_layers": 1, "weight_decay": 0.0, "hidden_dims": (128,)},
            {"question":"4. Implement a model without dropout but with L2 regularization (weight decay).", "name": "L2_Regularization", "activation": "relu", "batch_norm": True, "hidden_layers": 1, "weight_decay": 1e-4, "hidden_dims": (128,)},
            {"question":"5. Write a model with two hidden layers and softmax output", "name": "Two_HidLayers_SoftMax", "activation": "relu", "batch_norm": True, "hidden_layers": 2, "weight_decay": 0.0, "hidden_dims": (128, 64)},
        ]

    all_metrics = {}
    trained_models = {}

    # Train all models
    for config in steps_config:
       
        print(f"Question: {config['question']}")
         # config_name = config["name"]
        # if config_name not in ["BasicFeedforward","BatchNorm","LeakyReLU_Activation","L2_Regularization","Two_HidLayers_SoftMax"]: # Test
        #     continue
        model = SentimentFFN(
            len(vocab), 
            EMBED_DIM, 
            num_classes,
            activation=config["activation"],
            batch_norm=config["batch_norm"],
            hidden_layers=config["hidden_layers"],
            hidden_dims=config["hidden_dims"]
        )

        metrics, trained_model = train_model(config["name"], model, train_loader, val_loader, weight_decay=config["weight_decay"])
        all_metrics[config["name"]] = metrics
        trained_models[config["name"]] = trained_model

    # Evaluate and visualize results
    if all_metrics:
        plot_metrics(all_metrics, "models_comparison")
        print_summary_table(all_metrics)

    # Sample reviews for testing prediction
    sample_reviews = [
        "Movie was good for children only",
        "Absolutely hilarious! The hero slipped on a banana peel, and the villain got pied in the face. Oscar-worthy comedy!",
        "The plot was as confusing as a squirrel at a disco party, but I enjoyed every silly minute.",
        "I went to see a drama, but ended up watching a comedy of errors. The talking dog stole the show!",
        "If you like movies with ninjas, pirates, and a surprise appearance by a giant rubber duck, this one's for you.",
        "The movie was absolutely fantastic! I love it.",
        "I did not enjoy the movie. It was boring and too long.",
        "The acting was nice but the story not so interesting.",
        "An outstanding film with breathtaking animations and music.",
        "I enjoyed the movie very much",
        "Fine movie enjoyed all characters and scenes. Love the acting",
        "Terrible movie. Waste of popcorn, time and money.",
        "Ahh movie was in dual tone, I enjoyed sometime but not overall",
        "This movie was a complete disappointment from start to finish. The plot dragged on endlessly, with pointless scenes that added nothing to the story. The characters were poorly developed, making it hard to care about what happened to them. Even the special effects, which I had high hopes for, looked cheap and unconvincing. I kept waiting for something exciting to happen, but the film just kept getting worse. By the end, I was frustrated that I had wasted my time and money on such a boring and forgettable experience.",
        "From the opening scene to the final credits, this film was a delightful journey. The story was engaging and full of heart, with characters that felt real and relatable. The performances were top-notch, especially from the lead actor, who brought so much emotion to the role. The cinematography was stunning, capturing every moment beautifully. I laughed, I cried, and I left the theater feeling inspired. This is the kind of movie that stays with you long after you've seen it, and I can't recommend it enough.",
        "This movie was bananas! The villain was defeated by a rubber chicken and a dancing llama. I couldn't stop laughing.",
        ]

    if trained_models:
        review_comparison_table(sample_reviews, vocab, trained_models)
        plot_review_probabilities(sample_reviews, vocab, trained_models, "models_comparison")

        # Save and load model to test
        save_models(trained_models)
        loaded_models = load_models(steps_config, vocab, num_classes)
        if loaded_models:
            print("\nTesting Loaded Models ---")
            review_comparison_table(sample_reviews, vocab, loaded_models)
            plot_review_probabilities(sample_reviews, vocab, loaded_models, "models_comparison")
        else:
            print("\nModels (loaded_models) not found, cannot be loaded.")
    else:
        print("\n---Trained models not found.")

if __name__ == "__main__":
    main()
