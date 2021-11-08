from transformers import BertTokenizerFast, BertForSequenceClassification
from dataset import DBDCDataset
import argparse
from transformers import Trainer, TrainingArguments
import logging
import sys
import glob
from tqdm import tqdm
from utils import *
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='test_data_path', action='store', required=True)
    parser.add_argument('-o', dest='output_path', action='store', required=True)
    parser.add_argument('-m', dest='model', action='store', required=True)
    parser.add_argument('-h', dest='history_context', default = 0, action='store', required=True)
    args = parser.parse_args()

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=3).to('cuda')

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=False)

    test_files = glob.glob(args.test_data_path + "/*")

    for _, test_file in enumerate(tqdm(test_files)):

        data = prepare_input(test_file, tokenizer, history_context=args.history_context)

        outputs = model(**data)

        print(outputs)