import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, AutoConfig, get_scheduler, default_data_collator, DataCollatorWithPadding, DataCollatorForTokenClassification
from tqdm import tqdm
import random
import scipy
import math
from models.attack_models import InversionMLP, InversionPLM, InversionTransformer
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def bulid_dataloader_sentence(task_name='sst2', max_length=128, batch_size=32):
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of= None)
    if task_name in task_to_keys:
        raw_datasets = load_dataset("glue", task_name)
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        raw_datasets = load_dataset(task_name)
        sentence1_key, sentence2_key = ('text', None)
    padding =  False
    max_length = 128
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] if 'validation' in processed_datasets else processed_datasets['test']
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader
    
def bulid_dataloader_token(task_name=None, train_file=None, eval_file=None, batch_size=128, max_length=128):
    data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of= None
        )
    if task_name == None:
        data_files = {}
        data_files["train"] = train_file
        data_files["validation"] = eval_file
        extension =  train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        raw_datasets = load_dataset(task_name)
    
    # raw_datasets = load_dataset("glue", task_name)
    # sentence1_key, sentence2_key = task_to_keys[task_name]
    text_column_name = 'tokens'
    padding =  False
    max_length = 128
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["test"]
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    
    return train_dataloader, eval_dataloader

def dataloader2memory(dataloader, model, target_layer=3):
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    device = model.device
    for batch in dataloader:
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            # batch['output_hidden_states'] = True
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            input_ids = batch['input_ids'].to('cpu')
            attention_mask = batch['attention_mask'].to('cpu')
            target_hidden_states = outputs.hidden_states[target_layer].to('cpu')
            features.append({'hidden_states': target_hidden_states, 'input_ids': input_ids, 'attention_mask': attention_mask})
        pro_bar.update(1)
    return features
   
def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids

from rouge_score import rouge_scorer
import re

def rouge(input_ids, pred_ids, tokenizer):
    # input_ids (bsz, seq_len)
    # pred_ids (bsz, seq_len)
    batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in input_ids]
    batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in pred_ids]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # scores = scorer.score('The quick brown fox jumps over the lazy dog',
    #                   'The quick brown dog jumps on the log.')
    hit_cnt = 0
    total_cnt = 0    
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", " ", real_tokens).strip()
        real_tokens = ' '.join(real_tokens.split())
        pred_tokens = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", " ", pred_tokens).strip()
        pred_tokens = ' '.join(pred_tokens.split())
        
        rouge_score = scorer.score(real_tokens, pred_tokens)['rougeL'].fmeasure
        hit_cnt += rouge_score
        total_cnt += 1
    return hit_cnt, total_cnt


def train_inversion_model(train_dataloader, eval_dataloader, inversion_model_type='plm', inversion_epochs=5, inversion_lr=5e-5, inversion_topk=1, accelerator=None, output_dir=None):
    if inversion_model_type == 'plm':
        inversion_model = InversionPLM(config)
    elif inversion_model_type == 'mlp':
        inversion_model = InversionMLP(config)
    device = accelerator.device
    # inversion_model.to(device)
    inversion_model = accelerator.prepare(inversion_model)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in inversion_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in inversion_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=inversion_lr)

    total_step = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_step,
    )

    progress_bar = tqdm(range(total_step))
    
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    filter_tokens = list(set(special_tokens))
    
    # device = accelerator.device
    completed_steps = 0
    print('################# start train inversion model #################')
    
    best_attack = 0
    for epoch in range(inversion_epochs):
        for step, batch in enumerate(train_dataloader):            
            batch = {key:value.to(device) for key,value in batch.items()}
    
            target_hidden_states = batch['hidden_states']
            labels = batch['input_ids']
            labels[word_filter(labels, filter_tokens)]=-100
            
            attention_mask = batch['attention_mask']
            
            logits, loss = inversion_model(target_hidden_states, labels, attention_mask=attention_mask)
        
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('loss:{}'.format(loss.item()))

        if True:
            hit_cnt = [0 for _ in range(len(inversion_topk))]
            total_cnt = [0 for _ in range(len(inversion_topk))]
            # hit_cnt = 0
            # total_cnt = 0
            for batch in eval_dataloader:
                batch = {key:value.to(device) for key,value in batch.items()}
                
                target_hidden_states = batch['hidden_states']
                eval_label = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                pred_logits, preds = inversion_model.predict(target_hidden_states, attention_mask=attention_mask)
                
                valid_ids = attention_mask!=0
                valid_ids[word_filter(eval_label, filter_tokens)] = False
                eval_label = eval_label[valid_ids] 
                
                for idx, k in enumerate(inversion_topk):
                    preds = torch.topk(pred_logits, k=k)[1] 
                    preds = preds[valid_ids]

                    hit_cnt[idx] += (eval_label.unsqueeze(1) == preds).int().sum().item()
                    total_cnt[idx] += eval_label.shape[0]
            if (hit_cnt[0]/total_cnt[0])>=best_attack:
                best_attack = hit_cnt[0]/total_cnt[0]
                
                torch.save(inversion_model, f'{output_dir}/inversion_model.pt')
                print(f'save inversion model in {output_dir}/inversion_model.pt')
                
            for idx, k in enumerate(inversion_topk):
                print('attack top {} acc:{}'.format(k, hit_cnt[idx]/total_cnt[idx]))

def evaluate_knn(emb, eval_dataloader, tokenizer, topk=[1,5], device='cuda'):
    progress_bar = tqdm(range(len(eval_dataloader)))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    filter_tokens = list(set(special_tokens))
    
    # device = accelerator.device
    completed_steps = 0
    print('################# evaluate knn #################')
    
    attack_results = {}
    hit_cnt = [0 for _ in range(len(topk))]
    total_cnt = [0 for _ in range(len(topk))]
    rouge_hit_cnt = 0
    rouge_total_cnt = 0
    for batch in eval_dataloader:
        batch = {key:value.to(device) for key,value in batch.items()}
        target_hidden_states = batch['hidden_states']

        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0

        eval_label = batch['input_ids']
        valid_ids[word_filter(eval_label, filter_tokens)] = False
        eval_label = eval_label[valid_ids] # (samples)
        preds_feature = target_hidden_states
        bsz, seq_len, dim = preds_feature.shape
        ed = torch.cdist(preds_feature.view(-1, dim), emb, p=2.0).view(bsz, seq_len, -1) # (samples, embeddings)
        # ed = cosine_similarity(preds_feature, emb)
        for idx, k in enumerate(topk):
            candidate_token_ids_topk = torch.topk(ed, k, largest=False)[1] # (bsz, seq_len, topk)
            candidate_token_ids_topk = candidate_token_ids_topk[valid_ids] # (sample, topk)
            hit_cnt[idx] += (eval_label.unsqueeze(1) == candidate_token_ids_topk).int().sum().item()
            total_cnt[idx] += eval_label.shape[0]
        
        candidate_token_ids_top1 = torch.topk(ed, k=1, largest=False)[1].squeeze()
        candidate_token_ids_top1[valid_ids==0] = 1
        eval_label = batch['input_ids'].clone().detach()
        eval_label[valid_ids==0] = 1
        r_hit_cnt, r_total_cnt = rouge(eval_label, candidate_token_ids_top1, tokenizer)
        rouge_hit_cnt += r_hit_cnt
        rouge_total_cnt += r_total_cnt
        progress_bar.update(1)
    
    
    for idx, k in enumerate(topk):
        attack_results[f'knn-attack top {k}'] = hit_cnt[idx]/total_cnt[idx]
        # print('knn-attack top {} acc:{}'.format(k, hit_cnt[idx]/total_cnt[idx]))
    attack_results['knn-attack rouge'] = rouge_hit_cnt/rouge_total_cnt
    return attack_results

def evaluate_inversion_model(inversion_model, eval_dataloader, tokenizer,  inversion_topk=[1,5], device='cuda'):
    progress_bar = tqdm(range(len(eval_dataloader)))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    simple_tokens = []
    filter_tokens = list(set(special_tokens + simple_tokens))
    
    # device = accelerator.device
    completed_steps = 0
    print('################# start train inversion model #################')
    
    attack_results = {}
    hit_cnt = [0 for _ in range(len(inversion_topk))]
    total_cnt = [0 for _ in range(len(inversion_topk))]
    rouge_hit_cnt = 0
    rouge_total_cnt = 0
    for batch in eval_dataloader:
        batch = {key:value.to(device) for key,value in batch.items()}
        target_hidden_states = batch['hidden_states']
        eval_label = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        pred_logits, preds = inversion_model.predict(target_hidden_states, attention_mask=attention_mask)
        
        valid_ids = attention_mask!=0
        valid_ids[word_filter(eval_label, filter_tokens)] = False
        eval_label = eval_label[valid_ids] 
        
        for idx, k in enumerate(inversion_topk):
            preds = torch.topk(pred_logits, k=k)[1] 
            preds = preds[valid_ids]

            hit_cnt[idx] += (eval_label.unsqueeze(1) == preds).int().sum().item()
            total_cnt[idx] += eval_label.shape[0]
        
        candidate_token_ids_top1 = torch.topk(pred_logits, k=1)[1].squeeze()
        candidate_token_ids_top1[valid_ids==0] = 1
        eval_label = batch['input_ids'].clone().detach()
        eval_label[valid_ids==0] = 1
        r_hit_cnt, r_total_cnt = rouge(eval_label, candidate_token_ids_top1, tokenizer)
        rouge_hit_cnt += r_hit_cnt
        rouge_total_cnt += r_total_cnt
        progress_bar.update(1)
        
    for idx, k in enumerate(topk):
        attack_results[f'inversion-attack top {k}'] = hit_cnt[idx]/total_cnt[idx]
    attack_results['inversion-attack rouge'] = rouge_hit_cnt/rouge_total_cnt
    return attack_results

import os
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default='roberta-base',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default='conll2003',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default='finetune',
        help="The name of the dataset to use (via the datasets library).",
    )
    args = parser.parse_args()
    model_name = args.model_path
    task_name = args.task_name
    method_name = args.method_name
    
    if task_name == 'ag_news':
        batch_size = 32
    else:
        batch_size = 64
    
    target_layer=3

    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    
    
    from models.modeling_roberta_privacy import RobertaForSequenceClassification, RobertaForTokenClassification
    config.target_break = 0
            
    if task_name in ['conll2003', 'tner/ontonotes5']:
        model = RobertaForTokenClassification.from_pretrained(model_name, config=config)
        train_dataloader, eval_dataloader = bulid_dataloader_token(task_name, batch_size=batch_size)
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
        train_dataloader, eval_dataloader = bulid_dataloader_sentence(task_name, batch_size=batch_size)
    
    model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)
    
    print('load dataloader to memory')
    train_dataloader = dataloader2memory(train_dataloader, model, target_layer)
    eval_dataloader = dataloader2memory(eval_dataloader, model, target_layer)
    print('done')
    
    del model
    torch.cuda.empty_cache()
    
    device = accelerator.device
    topk = [1,5]
    
    os.makedirs('./logs/inversion/', exist_ok=True)
    
    for learning_rate in [5e-5]:
        device='cuda'
        epochs=10
        topk = [1,5]
        train_inversion_model(train_dataloader, eval_dataloader, 'plm', epochs, learning_rate, topk, accelerator, output_dir=model_name)
        inversion_model = torch.load(f'{model_name}/inversion_model.pt')
        inversion_results = evaluate_inversion_model(inversion_model, eval_dataloader, tokenizer, topk)
        print(inversion_results)
        if 'ontonotes' in task_name:
            f = open(f'./logs/inversion/ontonotes.txt', 'a')
        else:
            f = open(f'./logs/inversion/{task_name}.txt', 'a')
            
        f.write(f'{method_name}\n lr: {learning_rate}')
        for key,value in inversion_results.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\n') 
        f.close()