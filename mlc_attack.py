import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, AutoConfig, get_scheduler, default_data_collator, DataCollatorWithPadding, DataCollatorForTokenClassification
from tqdm import tqdm
import random
import scipy
import math
from models.attack_models import InversionPLMMLC
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from models.utils import token_hit

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
    
def train_mlc_model(train_dataloader, eval_dataloader, inversion_model_type='plm', inversion_epochs=5, inversion_lr=5e-5, inversion_topk=1, device='cuda'):
    if inversion_model_type == 'plm':
        inversion_model = InversionPLMMLC(config)
    
    inversion_model.to(device)
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
    simple_tokens = []
    filter_tokens = list(set(special_tokens + simple_tokens))
    
    # device = accelerator.device
    completed_steps = 0
    print('################# start train mlc model #################')
    
    best_performance = 0
    for epoch in range(inversion_epochs):
        for step, batch in enumerate(train_dataloader):
            # input_embeddings = embedding[batch['input_ids']].clone().detach()
            
            batch = {key:value.to(device) for key,value in batch.items()}
    
            target_hidden_states = batch['hidden_states']
            labels = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            logits, loss = inversion_model(target_hidden_states, labels, attention_mask=attention_mask)
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('loss:{}'.format(loss.item()))

        if True:
            hit_cnt = 0
            total_cnt = 0
            for batch in eval_dataloader:
                batch = {key:value.to(device) for key,value in batch.items()}
                
                target_hidden_states = batch['hidden_states']
                eval_label = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                pred_logits, batch_preds = inversion_model.predict(target_hidden_states, attention_mask=attention_mask)
                bsz, _ = batch_preds.shape
                batch_eval_label = batch['input_ids']
                for i in range(bsz):
                    preds = batch_preds[i].nonzero().squeeze().unsqueeze(0)
                    eval_label = batch_eval_label[i].unsqueeze(0)
                    temp_hit, temp_total = token_hit(eval_label, preds, tokenizer, special_tokens)
                    hit_cnt += temp_hit
                    total_cnt += temp_total
            
            if hit_cnt/total_cnt > best_performance:
                best_performance = hit_cnt/total_cnt
                torch.save(inversion_model, f'{model_name}/mlc_model.pt')
                print(f'model save to {model_name}/mlc_model.pt')
            
            print('attack acc:{}'.format(hit_cnt/total_cnt))
            
def evaluate_mlc_model(inversion_model, eval_dataloader, tokenizer):
    inversion_model.to(device)
    progress_bar = tqdm(range(len(eval_dataloader)))
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    simple_tokens = []
    filter_tokens = list(set(special_tokens + simple_tokens))
    completed_steps = 0
    print('################# start train mlc model #################')
    
    hit_cnt = 0
    total_cnt = 0
    results = {}
    for batch in eval_dataloader:
        batch = {key:value.to(device) for key,value in batch.items()}
        
        target_hidden_states = batch['hidden_states']
        eval_label = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        pred_logits, batch_preds = inversion_model.predict(target_hidden_states, attention_mask=attention_mask)
        bsz, _ = batch_preds.shape
        batch_eval_label = batch['input_ids']
        for i in range(bsz):
            preds = batch_preds[i].nonzero().squeeze().unsqueeze(0)
            eval_label = batch_eval_label[i].unsqueeze(0)
            temp_hit, temp_total = token_hit(eval_label, preds, tokenizer, special_tokens)
            hit_cnt += temp_hit
            total_cnt += temp_total
    results['mlc_attack'] = hit_cnt/total_cnt
    return results
    # print('attack acc:{}'.format(hit_cnt/total_cnt))

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
    
    for learning_rate in [5e-5]:
        device='cuda'
        epochs=20
        topk = 1
        
        train_mlc_model(train_dataloader, eval_dataloader, 'plm', epochs, learning_rate, topk, device)
        inversion_model = torch.load(f'{model_name}/mlc_model.pt')
        inversion_results = evaluate_mlc_model(inversion_model, eval_dataloader, tokenizer)
        if 'ontonotes' in task_name:
            f = open(f'./logs/inversion/ontonotes.txt', 'a')
        else:
            f = open(f'./logs/inversion/{task_name}.txt', 'a')
            
        f.write(f'MLC {method_name}\n')
        for key,value in inversion_results.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\n') 
        f.close()