import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
import NER_functions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from seqeval.metrics import f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras_preprocessing.sequence import pad_sequences

import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def main():
    data = pd.read_csv("Dataset/NER dataset.csv", encoding="latin1").fillna(method="ffill")
    getter = SentenceGetter(data)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    labels = [[s[2] for s in sentence] for sentence in getter.sentences]
    tag_values, tag2idx = NER_functions.get_tag_values(data)
    MAX_LEN = 104
    bs = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenized_texts_and_labels = [NER_functions.tokenize_and_preserve_labels(sent, labs, tokenizer) for sent, labs in
                                  zip(sentences, labels)]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    # Split the data into training, validation, and test sets
    train_inputs, temp_inputs, train_tags, temp_tags = train_test_split(input_ids, tags, random_state=2018,
                                                                        test_size=0.2)
    val_inputs, test_inputs, val_tags, test_tags = train_test_split(temp_inputs, temp_tags, random_state=2018,
                                                                    test_size=0.2)
    train_masks, temp_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.2)
    val_masks, test_masks, _, _ = train_test_split(temp_masks, temp_inputs, random_state=2018, test_size=0.2)

    # Convert the data into PyTorch tensors
    train_inputs = torch.tensor(train_inputs)
    val_inputs = torch.tensor(val_inputs)
    test_inputs = torch.tensor(test_inputs)

    train_tags = torch.tensor(train_tags)
    val_tags = torch.tensor(val_tags)
    test_tags = torch.tensor(test_tags)

    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    test_masks = torch.tensor(test_masks)

    # Create DataLoader for training, validation, and test sets
    train_data = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    test_data = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(tag2idx),
        output_attentions=False,
        output_hidden_states=False
    )

    model.cpu()
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    epochs = 3
    max_grad_norm = 1.0

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_values, validation_loss_values = [], []
    accuracy_values, validation_accuracy_values = [], []

    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_labels = b_labels.long()

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0].long()
            # Filter parameters that require gradients
            params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        # Evaluation on the validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)

        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if tag_values[l_i] != "PAD"]

        validation_accuracy = accuracy_score(pred_tags, valid_tags)
        validation_accuracy_values.append(validation_accuracy)

        accuracy_values.append(validation_accuracy)  # Store accuracy for plotting

        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print()

        # Evaluation on the test set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(test_dataloader)
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        test_tags = [tag_values[l_i] for l in true_labels
                     for l_i in l if tag_values[l_i] != "PAD"]
        test_accuracy = accuracy_score(pred_tags, test_tags)

        print("Test Accuracy: {}".format(test_accuracy))

    classification_rep = classification_report(test_tags, pred_tags)
    print("Classification Report BERT Model:\n", classification_rep)


if __name__ == "__main__":
    main()
