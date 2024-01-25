from model import build_transformer
from newdataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path


import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

# Huggingface datasets and tokenizers
#from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

import random
import numpy as np
import re
import time


import warnings
from tqdm import tqdm
import os
from pathlib import Path


import torchmetrics
from torch.utils.tensorboard import SummaryWriter



# Data set formater  new input output decoder:
def format_dataset(input_exp, target_exp):
    input_exp = input_vectorization(input_exp)
    target_exp = output_vectorization(target_exp)
    return ({"encoder_inputs": input_exp, "decoder_inputs": target_exp[:, :-1],}, target_exp[:, 1:])

def make_dataset(pairs):
    intr, fey_texts, sqampl_texts, t = zip(*pairs)
    fey_texts = list(fey_texts)
    sqampl_texts = list(sqampl_texts)
    dataset = tf.data.Dataset.from_tensor_slices((fey_texts, sqampl_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()





## Data preprocessing
#preprocessing for the amplitudes:

def prepro_feyn(data):

    for r in (('(', '('), (')', ')'), ('  ', ' '), (' e(', 'e(m_e,-1,' ),(' mu(', 'mu(m_mu,-1,') ,(' u(', ' u(m_u,2/3,'), (' d(', 'd(m_d,-1/3,'), (' t(', ' t(m_t,-1,') ,(' s(', 's(m_s,-1/3,'), (' tt(', ' tt(m_tt,-1,'), (' c(', 'c(m_c,2/3,'),(' b(', 'b(m_b,-1/3,'), ('Anti ', 'Anti,'), ('Off ', 'Off,'), ('  ', ' ') ):
        data = data.replace(*r)

    return data

#preprocessing for the squared amplitudes:
def prepro_squared_ampl(data):

    for r in (('*', '*'), (',', ' , '), ('*(', ' *( ') , ('([', '[ '), ('])', ' ]'), ('[', '[ '), (']', ' ]'), ('[ start ]', '[start]'), ('[ end ]', '[end]'), (' - ', ' -'), (' + ',' +' ) ,('/', ' / ') ,('  ', ' ')) :
        data = data.replace(*r)
    data = re.sub(r"\*(s_\d+\*s_\d+)", r"* \1", data)
    data = re.sub(r"\*(s_\d+\^\d+\*s_\d+)", r"* \1", data)
    data = re.sub(r"\*(m_\w+\^\d+\*s_\d+)", r"* \1", data)
    data = re.sub(r"(m_\w+\^\d+)", r" \1 ", data)
    data = data.replace('  ', ' ')


    return data

def max_len(sq_data):
    l = len(sq_data[sq_data.index(max(sq_data, key=len))].split())
    return l


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    #for item in ds:
    #   yield item[lang]
    for text  in ds:
        for token in text['translation'][lang].split():
            yield token.split()


def get_or_build_tokenizer(config, ds, lang):
    # giving a language it creates the  config['tokenizer_file']='../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer




def get_ds(config):
    # config['datasource']=fey_qed_order.txt, see config module.
    #with open(config['datasource'], 'r', encoding='utf-8') as f:
    with open('/content/data/DATA/fey_qed_order.txt', 'r', encoding='utf-8') as f:   
        lines = f.read().split('\n')
    text_pairs =[]

    for line in lines[: min(len(lines), len(lines)-1)]:
        intr, amp, sqamp, t  = line.split('>')
        #sqamp = "[start] " + sqamp + " [end]"

        text_pairs.append((intr, amp,sqamp, float(t) ))

    text_pairs = list(set(text_pairs))
    print('data size: ', len(text_pairs))

    text_pairs_prep = []
    for i in range(len(text_pairs)):
        string=""
        sqam=prepro_squared_ampl(text_pairs[i][2])
        for k in sqam:
            string+=k
        if len(string)<500:
            text_pairs_prep.append({'id':i,'translation':{'interaction':text_pairs[i][0], 'graph':prepro_feyn(text_pairs[i][1]), 'sqAmp':string, 't':text_pairs[i][3]}})

    text_pairs = text_pairs_prep
    feyn = [pair['translation'][config['lang_src']] for pair in text_pairs]
    sq_ampl= [pair['translation'][config['lang_tgt']] for pair in text_pairs]
    print ('Example amplitude:' ,sq_ampl[1])
    print( 'Maximum sequence length of Feynman diagram   :' ,max_len(feyn))
    print( 'Maximum sequence length of squared amplitudes:' ,max_len(sq_ampl))

    ds_raw = text_pairs
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
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
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

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

