from transformers import BertTokenizerFast, BertForSequenceClassification
from dataset import DBDCDataset
import argparse
from transformers import Trainer, TrainingArguments
import logging
import sys
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from datasets import load_metric
import wandb
from IPython import embed
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='training_data_path', action='store', required=True)
    parser.add_argument('-m', dest='model', action='store', required=True)
    parser.add_argument('-e', dest='epochs', action='store', default = 1, type=int, required=False)
    parser.add_argument('-c', dest='history_context', action='store', default = 0,type=int, required=False)
    args = parser.parse_args()



    model_saved_name = "./saved_models/"+args.model +"_"+str(args.epochs)+"epochs_"+str(args.history_context)+"historyContext"
    wandb.init(name = model_saved_name, config={"history_context": args.history_context})
    training_data_path = args.training_data_path
    model_name = args.model

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to('cuda')

    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

    dataset = DBDCDataset(training_data_path, tokenizer, history_context=args.history_context)

    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Splitting dataset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir=model_saved_name,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,
        warmup_steps=200,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=50,  # log & save weights each logging_steps
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()