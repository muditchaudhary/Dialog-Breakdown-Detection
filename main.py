from transformers import BertTokenizerFast, BertForSequenceClassification
from dataset import DBDCDataset
import argparse
from transformers import Trainer, TrainingArguments
import logging
import sys
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='training_data_path', action='store', required=True)
    parser.add_argument('-m', dest='model', action='store', required=True)
    parser.add_argument('-o', dest='saved_model_path', required=True)
    parser.add_argument('-e', dest='epochs', action='store', default = 1, type=int, required=False)
    parser.add_argument('-c', dest='history_context', action='store', default = 0,type=int, required=False)
    args = parser.parse_args()

    training_data_path = args.training_data_path
    model_name = args.model

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to('cuda')

    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

    train_dataset = DBDCDataset(training_data_path, tokenizer, history_context=args.history_context)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=50,  # log & save weights each logging_steps
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
    )

    trainer.train()

    model.save_pretrained(args.saved_model_path)
    tokenizer.save_pretrained(args.saved_model_path)