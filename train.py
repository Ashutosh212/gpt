import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


# def get_all_sentences(ds, lang):
#     for item in ds:
#         # Is item represnting pair here
#         yield item[lang]

def get_all_sentences(ds, lang):
    for item in ds:
        # Skip the pair if any sentence is empty
        if not item.get(lang):  # If the sentence is empty (None or empty string)
            continue
        yield item[lang]


# def get_or_build_tokenizer(config, ds, lang):
#     tokenizer_path = Path(config['tokenizer_file'].format(lang))
#     print(f"tokenizer path : {tokenizer_path}")
#     if not Path.exists(tokenizer_path):
#         tokenizer = Tokenizer(WordLevel(unk_token='unk'))
#         tokenizer.pre_tokenizer = Whitespace()
#         trainer = WordLevelTrainer(special_token = ["unk", "pad", "sos", "eos"], min_freequency = 2)

#         sentences = list(get_all_sentences(ds, lang))
#         tokenizer.train_from_iterator(sentences, trainer=trainer)
#         tokenizer.save(str(tokenizer_path))
    
#     else:
#         tokenizer = Tokenizer.from_file(str(tokenizer_path))

#     return tokenizer


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    print(f"tokenizer path : {tokenizer_path}")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
        sentences = list(get_all_sentences(ds, lang))

        tokenizer.train_from_iterator(sentences, trainer=trainer)
        cls_token_id = tokenizer.token_to_id("[CLS]")
        sep_token_id = tokenizer.token_to_id("[SEP]")
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset("csv" , data_files = "hindi_english_parallel.csv")

    ds_raw = ds_raw['train']
    # dataset = [{"english": example["english"], "hindi": example["hindi"]} for example in ds_raw]
    dataset = [
        {"english": example["english"], "hindi": example["hindi"]}
        for example in ds_raw
        if example["english"] and example["hindi"]  # Ensure neither value is None or empty
    ]
    for example in ds_raw:
        hindi_sentence = example['hindi']  
        english_sentence = example['english']  
        
        # Print the pair of sentences
        print("Hindi: ", hindi_sentence)
        print("English: ", english_sentence)
        print()  
        break  
    # print("Sample dataset:", dataset[:5])
    tokenizer_src = get_or_build_tokenizer(config, dataset, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset, config["lang_tgt"])

    # Splitting train-val

    train_ds_Size = int(0.9 * len(ds_raw))
    val_ds_Size = len(ds_raw) - train_ds_Size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_Size, val_ds_Size])



    # max_len_src = 0
    # max_len_tgt = 0

    # for example in dataset:
    #     src_text = example[config['lang_src']]
    #     tgt_text = example[config['lang_tgt']]

    #     if src_text and tgt_text:  # Ensure valid data
    #         src_ids = tokenizer_src.encode(src_text).ids
    #         tgt_ids = tokenizer_tgt.encode(tgt_text).ids  # FIXED: Use `tokenizer_tgt` here
            
    #         max_len_src = max(max_len_src, len(src_ids))
    #         max_len_tgt = max(max_len_tgt, len(tgt_ids))


    # print(f"Max length of source sentence: {max_len_src}")
    # print(f"Max length of target sentence: {max_len_tgt}")

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # train_ds = [sample for sample in train_ds if sample is not None]
    # val_ds = [sample for sample in val_ds if sample is not None]

    
    train_dataloader = DataLoader(train_ds , batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds , batch_size=1, shuffle=True)

    return  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            # (B, seq_len, vocab_size) -> (B * seq_len, vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


# Example call to the function
# get_ds(None)

