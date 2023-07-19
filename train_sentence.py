# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import os

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids
    

def evaluate_with_knn_attack(model, dataloader, metric, accelerator, filter_tokens=None, topk=5, target_layer=3):
    emb = model.roberta.embeddings.word_embeddings.weight
    model.eval()
    samples_seen = 0
    hit_cnt = 0
    total_cnt = 0
    
    for step, batch in enumerate(dataloader):
        batch['output_hidden_states'] = True
        with torch.no_grad():
            outputs = model(**batch)
        
        # evaluate
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        
        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0

        eval_label = batch['input_ids']
        valid_ids[word_filter(eval_label, filter_tokens)] = False
        eval_label = eval_label[valid_ids] # (samples)
        preds_feature = outputs.hidden_states[target_layer][valid_ids]
        ed = torch.cdist(preds_feature, emb, p=2.0) # (samples, embeddings)
        candidate_token_ids_topk = torch.topk(ed, topk, largest=False)[1] # (samples, topk)
        
        hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_topk).int().sum().item()
        total_cnt += eval_label.shape[0]
        
    eval_metric = metric.compute()
    eval_metric['knn_top{}'.format(topk)] = hit_cnt/total_cnt
    return eval_metric


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='sst2',
        help="The name of the glue task to train on.",
        # choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='roberta-base',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine_with_restarts",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        # action="store_true",
        default=0,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=3
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1
    )
    # CenterLoss
    parser.add_argument(
        "--w_cluster_away",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--w_cluster_close",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--recluster",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--cluster_subword",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--cluster_num",
        default=100,
        type=int,
    )
    args = parser.parse_args()
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    
    send_example_telemetry("run_glue_no_trainer", args)
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.task_name in task_to_keys:
            raw_datasets = load_dataset("glue", args.task_name)
        else:
            raw_datasets = load_dataset(args.task_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)


    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, model_input_names=["input_ids", "token_type_ids", "attention_mask", 'cluster_ids'])
    
    
    config.cluster_num = args.cluster_num

    # common hyperparameter
    config.target_layer = args.target_layer
    config.learning_rate = args.learning_rate
    config.recluster = args.recluster
    
    
    # hyperparameter for Privacy Loss
    config.eps = args.eps
    config.w_cluster_away = args.w_cluster_away
    config.w_cluster_close = args.w_cluster_close
    
    config.recluster = args.recluster
    

    from models.modeling_roberta_privacy import RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        if args.task_name in task_to_keys:
            sentence1_key, sentence2_key = task_to_keys[args.task_name]
        else:
            sentence1_key, sentence2_key = ('text', None)
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )

    train_dataset = processed_datasets["train"]
    if args.task_name in task_to_keys:
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    else:
        eval_dataset = processed_datasets["test"]

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer, 
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if (args.task_name is not None) and args.task_name in task_to_keys:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0
    
    # parameter for trianing
    knn_topk=10    
    previous_metric = 0
    tolerate = 10
    early_stop = tolerate
    privacy_target = {'sst2':0.012, 'ag_news':0.013}
    best_performance = 0
    config.target_break = 0
    
    
    import time
    from cluster_utils import run_cluster, redivide_cluster
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    simple_tokens = []
    filter_tokens = list(set(special_tokens + simple_tokens))
    
    if args.task_name == 'sst2':
        label_related_words = [['Ġpoorly', 'Ġpointless', 'Ġtires', 'Ġunnecessary', 'Ġbadly', 'Ġunimagin', 'thin', 'Ġlousy', 'Ġcrap', 'Ġjunk', 'Ġirritating', 'Ġplotting', 'Ġpreach', 'Ġbother', 'Ġinsult', 'Ġvap', 'Ġsour', 'Ġnowhere', 'Ġsuffers', 'Ġpit', 'Ġclueless', 'Ġdubbed', 'Ġhole', 'Ġunl', 'Ġexploitation', 'Ġhastily', 'Ġcopy', 'Ġlifeless', 'Ġscreaming', 'falls', 'Ġsleep', 'Ġvague', 'Ġdisguise', 'Ġwooden', 'Ġlackluster', 'poor', 'Ġgarbage', 'Ġthinks', 'Ġinsulting', 'Ġital', 'Ġexcruciating', 'Ġappears', 'Ġthrown', 'Ġloads', 'Ġmistake', 'Ġatro', 'Ġpr', 'Ġcob', 'Ġboat', 'Ġobligatory', 'Ġloose', 'Ġradical', 'Ġdrown', 'Ġsluggish', 'Ġredundant', 'Ġpity', 'Ġconco', 'Ġcostly', 'Ġlower', 'Ġrushed', 'Ġlaz', 'Ġquarter', 'Ġtransparent', 'Ġnonsense', 'Ġdisposable', 'Ġmeaningless', 'lost', 'Ġguess', 'Ġheads', 'Ġleak', 'Ġpool', 'Ġidiots', 'Ġexact', 'Ġwear', 'Ġbusy', 'Ġemptiness', 'Ġmil', 'Ġobscure', 'Ġdistant', 'control', 'Ġdisappointment', 'Ġturf', 'Ġsadd', 'Ġsplit', 'Ġstiff', 'ug', 'Ġpub', 'Ġchees', 'worst', 'skip', 'Ġetc', 'Ġruined', 'Ġnonexistent', 'Ġlimitations', 'Ġpopulated', 'Ġhist', 'Ġcage', 'Ġwarriors', 'Ġawkwardly', 'Ġarbitrary'], ['beaut', 'ple', 'Ġplayful', 'Ġbreathtaking', 'eng', 'Ġdetailed', 'Ġtender', 'Ġrive', 'Ġwonderfully', 'Ġdazzling', 'cle', 'Ġrewarding', 'Ġhearts', 'solid', 'Ġunpredictable', 'Ġhopeful', 'Ġupl', 'Ġtears', 'Ġelegant', 'Ġoutstanding', 'powerful', 'Ġvibrant', 'Ġpleasing', 'Ġrelief', 'Ġstartling', 'Ġvividly', 'Ġenjoyed', 'Ġsympathetic', 'Ġwond', 'Ġlocal', 'Ġahead', 'Ġstirring', 'Ġpotent', 'Ġguessing', 'Ġfluid', 'Ġflow', 'Ġhonesty', 'Ġirresistible', 'Ġsublime', 'Ġpolished', 'Ġopenness', 'Ġtechnology', 'Ġproud', 'Ġconstant', 'Ġbehold', 'scale', 'Ġinviting', 'Ġdelivering', 'Ġdawn', 'bring', 'touch', 'Ġlovers', 'Ġimmensely', 'Ġwhims', 'Ġgy', 'Ġsouls', 'Ġdelicious', 'Ġstrikes', 'Ġrapid', 'Ġembraces', 'Ġunderstands', 'Ġballot', 'Ġenduring', 'Ġencouraging', 'Ġrecovery', 'Ġpopular', 'Ġuncommon', 'Ġmiracle', 'Ġgently', 'Ġsly', 'Ġsensitivity', 'Ġforth', 'Ġsisters', 'Ġcompanion', 'Ġdraws', 'Ġwonders', 'Ġprofile', 'wise', 'Ġelegance', 'Ġcontribution', 'Ġexpressive', 'Ġstatus', 'Ġflashes', 'Ġrecommendation', 'Ġformat', 'Ġchord', 'Ġhay', 'Ġenormously', 'nice', 'Ġgrowth', 'Ġgorge', 'Ġsignificance', 'Ġliberating', 'Ġfinely', 'Ġcheek', 'color', 'Ġconfidence', 'Ġclinic', 'Ġsway', 'Ġties']]
    elif args.task_name == 'ag_news':
        label_related_words = [['stocks', 'aspx', '=/', 'http', '=', 'www', '="', '://', 'full', 'ĠWireless', 'Ġquarterly', 'ĠAirways', 'Ġsoftware', 'Ġconsumer', 'ĠMicrosoft', 'ĠSecurities', 'Ġinflation', 'ĠStores', 'ĠIBM', 'ĠMonetary', 'Ġcarrier', 'Ġmortgages', 'OR', 'ĠMortgage', 'ĠOv', 'Ġlender', 'Ġyen', 'ĠQuarterly', 'ĠTreasury', 'Nik', 'largest', 'ĠGrowth', 'ĠOPEC', 'Ġconsumers', 'ĠLower', 'ĠSprint', 'ĠIntel', 'NYSE', 'Ġestimates', 'Ġwireless', 'ĠSystems', 'ĠIPO', 'ĠMarsh', 'Trump', 'ĠRouse', 'Ġexports', 'Ġattendants', 'Google', 'EU', 'Ġretailers', 'ĠDollar', 'ĠMcL', 'Oracle', 'ĠBrands', 'Ġbroker', 'Ġsecurities', 'Ġhedge', 'Ġinventory', 'ĠCisco', 'ĠInvestors', 'ĠComputer', 'ĠGoldman', 'ĠAnimation', 'ĠSears', 'ĠGlobal', 'KK', 'Delta', 'OB', 'Microsoft', 'ĠFoods', 'chip', 'PK', 'ĠIMF', 'Ġasbestos', 'ĠMutual', 'ĠSachs', 'ĠEconomic', 'Ġinvent', 'ĠMond', 'ĠOrganization', 'Ġarthritis', 'ĠYahoo', 'Quote', 'Ġairlines', 'Ġlending', 'ĠAccounting', 'ny', 'HP', 'UAL', 'Ġacquisitions', 'ĠAbbey', 'Ġgrocer', 'Econom', 'Ġtech', 'Ġequity', 'ĠConsumer', 'ĠEconomy', 'ĠDepot', 'Ġloans', 'Ġgasoline'], ['ĠMicrosoft', 'Ġsoftware', '=', 'ĠiPod', 'ĠWindows', 'ĠYahoo', 'Microsoft', 'ĠMozilla', 'ĠLinux', 'ĠWireless', '="', 'ĠIBM', 'Ġq', 'info', 'sym', 'ĠServer', 'ĠCisco', 'http', 'ĠNASA', 'ĠIntel', 'NASA', 'ĠFirefox', 'ĠAMD', 'Ġprocessor', 'Ġwireless', '\\\\', 'www', 'ĠDesktop', 'ĠComputer', 'HP', 'ĠSPACE', 'Ġscientists', 'ĠHP', 'ĠIE', 'ĠHubble', 'Ġcomputing', 'uk', 'ĠXP', 'Ġdesktop', 'Ġworm', 'ĠIP', 'ĠMicro', 'ĠSoftware', 'ĠTelescope', 'ĠSolar', 'Ġbrowser', 'Ġserver', 'Google', 'Ġbeta', 'ĠSystems', 'ĠeBay', 'ĠDS', 'ĠMS', 'ĠGenesis', 'ĠXbox', 'Ġstorage', 'ĠWi', 'Ġdevices', 'Intel', 'Quote', 'source', 'Ġspam', '://', 'Ġhackers', 'ĠMessenger', 'ĠiTunes', 'Ġservers', 'ĠSans', 'ĠPCs', 'hel', 'ver', '="#', '6666', 'arial', 'Ġspacecraft', 'Fi', 'Ġapplication', 'ĠBlackBerry', 'ĠGmail', 'Ġtools', 'peer', 'ĠSkype', 'ĠAdobe', 'Ġbroadband', 'Ġconsumer', 'Ġdownload', 'ĠExplorer', 'ĠStorage', 'Ġenterprise', 'ĠNokia', 'Ġconsumers', 'ĠSaturn', 'ĠScientists', 'Linux', 'ĠJava', 'ĠLCD', 'Info', 'Ġspecies', 'ĠHalo', 'ĠUnix'], ['Ġquarterback', 'Sports', 'Ġtouchdown', 'ĠPacers', 'Ġinning', 'ĠPrix', 'Ġstriker', 'Ġtouchdowns', 'ĠEagles', 'Ġpreseason', 'Ġcricket', 'ĠNotre', 'Ġmedal', 'Ġpitcher', 'ĠKnicks', 'ĠChampions', 'Ġcoaching', 'ĠCavaliers', 'ĠTrophy', 'ĠNASCAR', 'ĠBroncos', 'Ġhomer', 'ĠBraves', 'ĠPhillies', 'Ġlinebacker', 'ĠChargers', 'ĠStadium', 'Ġoutfielder', 'ĠMVP', 'Ġunbeaten', 'ĠRedskins', 'Ġrookie', 'ĠRavens', 'NBA', 'ĠSteelers', 'ĠBlackburn', 'ĠFalcons', 'ĠColts', 'ĠRyder', 'Ġchampionship', 'ĠKobe', 'NL', 'ĠPistons', 'ĠBayern', 'ĠValencia', 'ĠNuggets', 'Ġcaptain', 'ĠTimberwolves', 'ĠMariners', 'finals', 'ĠAstros', 'Ġsemifinals', 'Ġqualifier', 'ĠUEFA', 'ĠFul', 'ĠWenger', 'ĠSerie', 'ĠCowboys', 'ĠMasters', 'ĠSPORTS', 'ĠReds', 'ĠChampionships', 'ĠClippers', 'ĠMarlins', 'ĠAdrian', 'ĠChelsea', 'Neal', 'ĠNCAA', 'ĠLakers', 'ĠTrafford', 'ĠNHL', 'ĠVikings', 'Ġbullpen', 'ĠBlazers', 'ĠSeahawks', 'Ġquarterbacks', 'ĠFIFA', 'ĠPatriots', 'ĠGators', 'ĠDame', 'Ġinnings', 'ĠCeltics', 'ĠWizards', 'ĠAthletics', 'ĠTournament', 'ĠEverton', 'ĠKl', 'ĠIndies', 'ĠOwen', 'ĠMets', 'ĠMut', 'Ġundefeated', 'ĠCubs', 'Ġbaseman', 'ĠBundesliga', 'ĠAngels', 'ĠJuventus', 'yard', 'ĠMLS', 'ĠRicky'], ['Ġmilitants', 'ĠAfghan', 'ĠHamas', 'Ġnuclear', 'ĠGaza', 'Ġcleric', 'Ġhostages', 'ĠAra', 'ĠAriel', 'ĠKabul', 'Qaida', 'ĠSunni', 'ĠMosul', 'ĠSaddam', 'Ġmosque', 'ĠIraqis', 'ĠMush', 'ĠShiite', 'Ġenrichment', 'Ġdisarm', 'ĠAbbas', 'ĠTaliban', 'ĠPalestinians', 'ĠSamar', 'Ġkidn', 'Ġbomber', 'ĠSharon', 'Palest', 'Ġsuicide', 'ĠKashmir', 'ĠRebels', 'ĠSyrian', 'Qaeda', 'ĠNaj', 'ĠNepal', 'ĠSinai', 'Ġinsurgent', 'Ġpolicemen', 'ĠRw', 'ĠAbu', 'Ġlandsl', 'ĠMyanmar', 'ĠMahmoud', 'ite', 'Ġconvoy', 'Ġbombing', 'Ġinsurgents', 'ĠHaiti', 'ĠPutin', 'Ġmilitias', 'ĠThatcher', 'Ġwounding', 'ĠMb', 'Ġinsurgency', 'ĠGuantanamo', 'ĠImam', 'ĠLebanese', 'Ġhumanitarian', 'Ġgenocide', 'ĠAfghans', 'ĠYugoslav', 'ĠIslam', 'ĠJordanian', 'ĠKilled', 'EU', 'Israeli', 'Sad', 'ĠHussein', 'ĠFein', 'ĠArist', 'ĠKurdish', 'Ġassassinated', 'Ġsurvivors', 'ĠSyria', 'ĠViktor', 'ĠNep', 'ĠJazeera', 'ĠSinn', 'Ġparamilitary', 'ĠPalestine', 'ĠLaden', 'Ġrepublic', 'Ġgrenade', 'ĠAtomic', 'Gaza', 'ĠDamascus', 'ĠNuclear', 'ĠHaitian', 'Ġkidnapping', 'ĠColin', 'ĠStraw', 'Syria', 'Ġembargo', 'ĠIslamist', 'ĠJewish', 'Ġdetainees', 'ĠEgyptian', 'Ġasylum', 'Ġprisoner', 'ĠLebanon']]
        

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.recluster:
            print('clustering......')
            config.w_noise = 0
            config.target_break = 1
            start_time = time.time()
            # token2cluster, cluster_center = run_cluster(model, train_dataloader, tokenizer, label_related_words, config.target_layer, config.cluster_num, cluster_method='kmeans')
            token2cluster, cluster_center, token_embeddings, sample2vocab = run_cluster(model, train_dataloader, tokenizer, label_related_words, config.target_layer, config.cluster_num, cluster_method='kmeans')
            token2cluster, cluster_center = redivide_cluster(label_related_words=label_related_words, cluster_center=cluster_center, tokenizer=tokenizer, token2cluster=token2cluster, token_embeddings=token_embeddings, embeddings_to_vocabulary=sample2vocab, cluster_num=config.cluster_num)
            model.privacy_loss.cluster_embedding.weight.data = cluster_center.type_as(model.privacy_loss.cluster_embedding.weight.data)
            config.w_noise = 1
            config.target_break = 0
            print('clustering......done! cost {} seconds'.format(time.time()-start_time))

        if early_stop == 0:
            break 
        
        model.train()
        total_loss = 0
        total_task_loss = 0
        total_privacy_loss = 0
        for step, batch in enumerate(train_dataloader):
            if config.recluster:
                batch['cluster_ids'] = torch.tensor([[token2cluster[ids.item()] for ids in batch_ids  ] for batch_ids in batch['input_ids']], device=batch['input_ids'].device)
            outputs = model(**batch)
            task_loss = outputs.loss
            privacy_loss_dict = outputs.privacy_loss
            privacy_loss = 0
            # if (step+1) > len(train_dataloader)/2:
            if privacy_loss_dict != None:
                for loss_item in privacy_loss_dict.values():
                    privacy_loss += loss_item
            loss = task_loss + privacy_loss    

            total_loss += loss.detach().float()
            total_task_loss += task_loss.detach().float()
            total_privacy_loss += privacy_loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            progress_bar.set_description('task:{:.5}|pri:{:.5}'.format(total_task_loss/(step+1), total_privacy_loss/(step+1)))
            
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
        
        eval_metric = evaluate_with_knn_attack(model, eval_dataloader, metric, accelerator, target_layer=args.target_layer, topk=knn_topk, filter_tokens=filter_tokens)
        

        if eval_metric['knn_top{}'.format(knn_topk)] <= privacy_target[args.task_name]:
            if eval_metric['accuracy'] > best_performance:
                best_performance = eval_metric['accuracy']
                
                output_dir = f'{args.output_dir}/epoch{epoch}'
                os.makedirs(output_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                with open(os.path.join(output_dir, "all_results.json"), "w") as f:
                    json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
            
        
        if previous_metric == eval_metric['accuracy']:
            early_stop -= 1
            
        else:
            early_stop = tolerate
        previous_metric = eval_metric['accuracy']

        logger.info(f"epoch {epoch}: {eval_metric}")
        progress_bar.set_description('acc:{:.4}'.format(eval_metric['accuracy']))
        
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    
    
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
   

if __name__ == "__main__":
    main()