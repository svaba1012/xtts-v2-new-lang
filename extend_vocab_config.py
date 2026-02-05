import argparse
from tokenizers import Tokenizer
import os
import pandas as pd
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import json
import torch

def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    # Load both the json files, take the union, and store it
    json1 = json.load(open(os.path.join(old_tokenizer, 'vocab.json'), encoding='utf-8'))
    json2 = json.load(open(os.path.join(new_tokenizer, 'vocab.json'), encoding='utf-8'))

    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Add words from second tokenizer
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Make the directory if necessary
    os.makedirs(save_dir, exist_ok=True)

    # Save the vocab
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)

    # Merge the two merges file. Don't handle duplicates here
    # Concatenate them, but ignore the first line of the second file
    os.system('cat {} > {}'.format(os.path.join(old_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(new_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))


def extend_tokenizer(args):
    
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")

    # save seperately vocab, merges
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)

    # train new tokenizer
    traindf = pd.read_csv(args.metadata_path, sep="|")
    texts = traindf.text.to_list()

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=[f"[{args.language}]"], vocab_size=args.extended_vocab_size)
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    new_tokenizer.add_special_tokens([f"[{args.language}]"])

    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)

    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )

    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(merged_tokenizer_path, 'merges.txt'))
    tokenizer.add_special_tokens([f"[{args.language}]"])

    tokenizer.save(os.path.join(root, "vocab.json"))

    os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

def adjust_config(args):
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["languages"] += [args.language]
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
def adjust_embeddings():
    
    # Load the checkpoint
    checkpoint = torch.load(os.path.join(args.output_path, "XTTS_v2.0_original_model_files/model.pth"), map_location="cpu")
    print(checkpoint.keys())
    state_dict = checkpoint["model"]

    # Resize embeddings
    old_emb = state_dict["gpt.text_embedding.weight"]
    new_vocab_size = 8464  # your new vocab
    new_emb = torch.nn.Parameter(torch.randn(new_vocab_size, old_emb.size(1)))  # random init
    new_emb[:old_emb.size(0), :] = old_emb.data
    state_dict["gpt.text_embedding.weight"] = new_emb

    # Text head
    old_head = state_dict["gpt.text_head.weight"]
    new_head = torch.nn.Parameter(torch.randn(new_vocab_size, old_head.size(1)))
    new_head[:old_head.size(0), :] = old_head.data
    state_dict["gpt.text_head.weight"] = new_head

    # Text head bias
    old_bias = state_dict["gpt.text_head.bias"]
    new_bias = torch.nn.Parameter(torch.zeros(new_vocab_size))
    new_bias[:old_bias.size(0)] = old_bias.data
    state_dict["gpt.text_head.bias"] = new_bias

    # Save back
    torch.save({"model": state_dict}, "model_resized.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, required=True, help="")
    parser.add_argument("--metadata_path", type=str, required=True, help="")
    parser.add_argument("--language", type=str, required=True, help="")
    parser.add_argument("--extended_vocab_size", default=2000, type=int, required=True, help="")

    args = parser.parse_args()

    extend_tokenizer(args)
    adjust_config(args)
    adjust_embeddings()