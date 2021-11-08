import glob
import json
def prepare_input(file, tokenizer, max_length=512, history_context = 0):
    dialog = []
    print("Preparing data")
    data = json.load(open(file))
    turns = data["turns"]

    all_turn_utterances = []
    for turn in turns:
        all_turn_utterances.append(turn["utterance"].lower())

    for i, turn in enumerate(turns):
        if (turn["speaker"] == "S" and not turn["annotations"] == []):

            current_dialog = turn["utterance"].lower()
            dialog_history = all_turn_utterances[i - (history_context if history_context < i else i):i]
            dialog_history = " ".join(dialog_history)
            if (history_context > 0):
                dialog = dialog_history + "<SEP>" + current_dialog
            else:
                dialog = current_dialog
            dialog.append(dialog)

    tokenized_dialogs = tokenizer(dialog, truncation=True, padding=True, max_length=max_length)

    return tokenized_dialogs
