from transformers import BertTokenizerFast, BertForSequenceClassification
from dataset import DBDCDataset
import argparse
from transformers import Trainer, TrainingArguments
import logging
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from model import *

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def calculate_accuracy(ground_truth, predictions):
    true_positives = 0
    true_negatives = 0

    for true, pred in zip(ground_truth, predictions):
        if (pred > 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass

    return (true_positives + true_negatives) / len(ground_truth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='training_data_path', action='store', required=True)
    parser.add_argument('-o', dest='saved_model_path', required=True)
    parser.add_argument('-b', dest='batch_size', required=True)
    parser.add_argument('-d', dest='hidden_size', required=True)
    parser.add_argument('-l', dest='lstm_layers', required=True)
    parser.add_argument('-e', dest='epochs', action='store', default = 1, type=int, required=False)
    parser.add_argument('-c', dest='history_context', action='store', default = 0,type=int, required=False)
    args = parser.parse_args()

    model = DialogClassifier(args)

    training_data_path = args.training_data_path
    training_set = DBDCDataset(training_data_path, args.history_context)

    training_loader = DataLoader(training_set, batch_size=args.batch_size)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):

        predictions = []
        model.train()

        for x_batch, y_batch in training_loader:
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)

            y_pred = model(x)

            loss = criterion(y_pred, y)



            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            predictions += list(y_pred.squeeze().detach().numpy())


        #train_accuary = calculate_accuracy(y_train, predictions)

        print("Epoch: %d, loss: %.5f" % (
        epoch + 1, loss.item()))

