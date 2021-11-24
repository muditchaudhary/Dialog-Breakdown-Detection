from transformers import BertTokenizerFast, BertForSequenceClassification
from dataset import DBDCDataset
import argparse
from transformers import Trainer, TrainingArguments
import logging
import sys
import glob
from tqdm import tqdm
from utils import *
from IPython import embed
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

label2idx = {
    "X":0,
    "T":1,
    "O":2
}

idx2label={
    0:"X",
    1:"T",
    2:"O"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='test_data_path', action='store', required=True)
    parser.add_argument('-o', dest='output_path', action='store', required=True)
    parser.add_argument('-m', dest='model_name', action='store', required=True)
    parser.add_argument('--history_context', default = 0, type= int)
    args = parser.parse_args()

    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=3).to('cuda')

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=False)

    test_files = glob.glob(args.test_data_path + "/*")

    for _, test_file in enumerate(tqdm(test_files)):

        test_json = json.load(open(test_file))

        data = prepare_input(test_json, tokenizer, history_context=args.history_context).to('cuda')

        outputs = model(**data)[0].softmax(1)


        output_json = {'dialogue-id': test_json["speaker-id"]}
        output_turns = []

        idx = 0
        for turn in test_json["turns"]:
            if turn["speaker"]=="S":
                this_turn_dict = {"turn-index": turn["turn-index"]}
                label = [
                    {
                        "breakdown": idx2label[outputs[idx].argmax().item()],
                        "prob-O": outputs[idx][2].item(),
                        "prob-T": outputs[idx][1].item(),
                        "prob-X": outputs[idx][0].item()

                    }
                ]
                this_turn_dict["labels"] = label
                output_turns.append(this_turn_dict)
                idx+=1

        output_json["turns"]=output_turns

        output_filename = test_file.split("/")[-1]
        output_filename = output_filename.split(".")[0]+".labels.json"

        with open(args.output_path+"/"+output_filename, "w") as f:
            json.dump(output_json, f, indent=4)