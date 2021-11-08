import glob
import json
def prepare_input(data, tokenizer, max_length=512, history_context = 0):
    processed_dialogs = []
    turns = data["turns"]

    all_turn_utterances = []
    for turn in turns:
        all_turn_utterances.append(turn["utterance"].lower())

    for i, turn in enumerate(turns):
        if (turn["speaker"] == "S"):

            current_dialog = turn["utterance"].lower()
            dialog_history = all_turn_utterances[i - (history_context if history_context < i else i):i]
            dialog_history = " ".join(dialog_history)
            if (history_context > 0):
                dialog = dialog_history + " [SEP] " + current_dialog
            else:
                dialog = current_dialog
            processed_dialogs.append(dialog)

    tokenized_dialogs = tokenizer(processed_dialogs, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    return tokenized_dialogs
