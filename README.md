# DeepLearning-Indepth

- Deep Learning Foundation
- CNN, RNN and LSTM
- Preparing unstructured Data for modeling
- Transformer Architecture and Evolution
- Build a Large Language model from Scratch
- Pydantic for LLM output Formating
- HuggingFace
- 4 project with deployment on CNN, RNN, and LLM
- Agents and RAG
- LangChain
- LangGraph
- LangSmith
- OpenAI
- CrewAI
- AnthopicAI
- LlamaIndex
- Ollama
- Gemini
- MCPs
- Agentic AI
- FastAPI for deployment
- Experiment Tracking: MLflow, Evidently, Grafana
- Containerization and Virtualization: Docker and Kubernetes
- AWS Cloud; Deployment with AWS Cloud, Amazon Sagemaker 


```python
#!/usr/bin/env python3
"""
train_text_classification.py

Fine-tune a transformer for single-label text classification with Hugging Face Transformers + Datasets.
Supports: automatic dataset column detection, tokenization, Trainer training loop, evaluation, saving, push-to-hub.

Example:
  python train_text_classification.py \
    --model_name_or_path distilbert-base-uncased \
    --dataset glue --dataset_config_name sst2 \
    --output_dir ./outputs/distilbert-sst2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --push_to_hub
"""

import argparse
import logging
import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune transformer for text classification")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset", type=str, default="glue", help="Dataset identifier or local path")
    parser.add_argument("--dataset_config_name", type=str, default="sst2", help="Dataset config (e.g., sst2 for glue)")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Optional: repo name on HF Hub")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision if available")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'auto'")
    return parser.parse_args()


def find_text_and_label_columns(dataset) -> (str, str):
    """
    Heuristic to find which columns correspond to text and label.
    Returns (text_column, label_column)
    """
    sample = dataset["train"][0]
    # candidate text columns
    text_candidates = ["text", "sentence", "review", "article", "content"]
    label_candidates = ["label", "labels", "stars"]

    text_column = None
    label_column = None
    for c in text_candidates:
        if c in sample:
            text_column = c
            break
    if text_column is None:
        # fallback: choose first string column
        for k, v in sample.items():
            if isinstance(v, str):
                text_column = k
                break
    for c in label_candidates:
        if c in sample:
            label_column = c
            break
    if label_column is None:
        # fallback: first int column
        for k, v in sample.items():
            if isinstance(v, int):
                label_column = k
                break
    if text_column is None or label_column is None:
        raise ValueError(f"Couldn't infer text/label columns. Sample keys: {list(sample.keys())}")
    return text_column, label_column


def preprocess_function(examples, tokenizer, text_column: str, max_length: int):
    texts = examples[text_column]
    # tokenizer can handle batch input
    return tokenizer(texts, truncation=True, max_length=max_length)


def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    # For single-label classification
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("Loading dataset %s (%s)", args.dataset, args.dataset_config_name)
    raw_datasets = load_dataset(args.dataset, args.dataset_config_name) if args.dataset_config_name else load_dataset(args.dataset)

    text_column, label_column = find_text_and_label_columns(raw_datasets)
    logger.info("Identified text column: %s, label column: %s", text_column, label_column)

    # Build label mapping
    # If labels are ints starting at 0, we can use num_labels directly
    # Otherwise, we build a list of unique labels from train
    unique_labels = sorted(set(raw_datasets["train"][label_column])) if "train" in raw_datasets else sorted(set(raw_datasets[list(raw_datasets.keys())[0]][label_column]))
    num_labels = len(unique_labels)
    logger.info("Number of labels: %d", num_labels)

    # Load tokenizer & model
    logger.info("Loading tokenizer and model from %s", args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # If tokenizer has more tokens than model embeddings, resize embeddings
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info("Resizing model embeddings to %d tokens", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize datasets
    logger.info("Tokenizing datasets (text col = %s)", text_column)
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer, text_column=text_column, max_length=args.max_length),
        batched=True,
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in [text_column, label_column]],
    )

    # Rename label column to "labels" expected by Trainer
    tokenized_datasets = tokenized_datasets.map(lambda examples: {"labels": examples[label_column]}, batched=True, remove_columns=[label_column])

    train_dataset = tokenized_datasets["train"] if "train" in tokenized_datasets else tokenized_datasets[list(tokenized_datasets.keys())[0]]
    eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else (tokenized_datasets.get("test", None))

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        seed=args.seed,
        load_best_model_at_end=True if args.evaluation_strategy != "no" else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn if eval_dataset is not None else None,
    )

    # Resume training if requested
    resumed = False
    if args.resume_from_checkpoint:
        ckpt = args.resume_from_checkpoint
        if ckpt == "auto":
            ckpt = trainer._load_from_checkpoint()  # may be None
        if ckpt:
            logger.info("Resuming training from checkpoint: %s", ckpt)
            trainer.train(resume_from_checkpoint=ckpt)
            resumed = True

    if not resumed:
        logger.info("Starting training")
        trainer.train()

    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if eval_dataset is not None:
        logger.info("Running final evaluation")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("Evaluation results: %s", metrics)

    if args.push_to_hub:
        logger.info("Pushing model to the Hugging Face Hub")
        trainer.push_to_hub(**({"commit_message": "Push model via training script"} if args.hub_model_id else {}))

    logger.info("Done.")


if __name__ == "__main__":
    main()
```