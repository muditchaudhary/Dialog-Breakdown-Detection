import torch
import glob
import json


label2idx = {
    "X":0,
    "T":1,
    "O":2
}

class DBDCDataset(torch.utils.Data.Dataset):

    def __init__(self, datapath, tokenizer, history_context = 0, max_length = 512):
        self.tokenizer = tokenizer
        self.dialog = []
        self.labels = []
        self.prepare_data()
        self.tokenized_dialogs = tokenizer(self.dialog, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.tokenized_dialogs.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

    def prepare_data(self):
        training_files = glob.glob(self.datapath + "/*")

        for training_file in training_files[:1]:
            print(training_file)
            data = json.load(open(training_file))
            turns = data["turns"]

            all_turn_utterances = []
            for turn in turns:
                all_turn_utterances.append(turn["utterance"].lower())

            for i, turn in enumerate(turns):
                if (turn["speaker"] == "S" and not turn["annotations"] == []):

                    current_dialog = turn["utterance"].lower()
                    dialog_history = all_turn_utterances[i - (self.history_context if self.history_context < i else i):i]
                    dialog_history = " ".join(dialog_history)
                    if (self.history_context > 0):
                        dialog = dialog_history + "<SEP>" + current_dialog
                    else:
                        dialog = current_dialog
                    self.dialog.append(dialog)
                    self.labels.append(label2idx[turn["annotations"]])