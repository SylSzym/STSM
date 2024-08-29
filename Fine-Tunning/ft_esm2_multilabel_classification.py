import pandas as pd
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from transformers import TrainingArguments, AutoTokenizer, EsmForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataProcessor:

    def __init__(self, data_train, data_eval, tokenizer_name, batch_size):
        self.data_train = data_train
        self.data_eval = data_eval
        self.tokenizer = tokenizer_name
        self.batch_size = batch_size

    def define_labels(self):
        not_chosen_columns = self.data_train.loc[:, ~self.data_train.columns.str.startswith('GO')]
        # Select label columns
        label_columns = [col for col in self.data_train.columns if col not in not_chosen_columns]

        # Create a new DataFrame containing only the selected label columns
        df_labels_train = self.data_train[label_columns]
        df_labels_eval = self.data_eval[label_columns]

        # Convert the label columns to lists for each row
        labels_list_train = df_labels_train.values.astype(np.float32).tolist()
        labels_list_eval = df_labels_eval.values.astype(np.float32).tolist()

        # prepare sequences - replace rare aminoacid
        self.data_train['motif'] = self.data_train['motif'].apply(lambda x: x.replace("[UZOB]", "X"))
        self.data_eval['motif'] = self.data_eval['motif'].apply(lambda x: x.replace("[UZOB]", "X"))

        train_seq = self.data_train['motif'].tolist()
        train_labels = labels_list_train

        eval_seq = self.data_eval['motif'].tolist()
        eval_labels = labels_list_eval
        return train_labels, eval_labels, train_seq, eval_seq, label_columns

    def calculate_embeddings(self):
        train_labels, eval_labels, train_seq, eval_seq, label_columns = self.define_labels()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, do_lower_case=False, max_length=512)
        train_encodings = tokenizer(train_seq, padding="max_length", truncation=True, max_length=512)
        eval_encodings = tokenizer(eval_seq, padding="max_length", truncation=True, max_length=512)
        return train_encodings, eval_encodings


class ModelTrainer:

    def __init__(self, model, num_labels, train_dataloader, validation_dataloader, tokenizer_name, eval_data,
                 thresholds, epochs, batch_size, label_columns):
        self.model = model
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.tokenizer = tokenizer_name
        self.eval_data = eval_data
        self.thresholds = thresholds
        self.label_columns = label_columns

    def train(self):
        train_loss_set = []
        val_f1_accuracy_list, val_flat_accuracy_list, training_loss_list, epochs_list = [], [], [], []

        for _ in trange(self.epochs, desc="Epoch "):
            # Training

            # Set our model to training mode
            self.model.train()

            # Tracking variables
            tr_loss = 0  # running loss
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(self.train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]
                loss_func = BCEWithLogitsLoss()
                # convert labels to float for calculation
                loss = loss_func(logits.view(-1, self.num_labels),
                                 b_labels.type_as(logits).view(-1, self.num_labels))

                train_loss_set.append(loss.item())

                # Backward pass
                loss.backward()

                # scheduler.step()
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))
            training_loss_list.append(tr_loss / nb_tr_steps)

    def evaluate(self):
        self.model.eval()
        # tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, do_lower_case=False, max_length=512)
        self.eval_data['motif'] = self.eval_data['motif'].apply(lambda x: x.replace("[UZOB]", "X"))
        # eval_seq = self.eval_data['motif'].tolist()
        # eval_labels = self.eval_data.loc[:, self.eval_data.columns.str.startswith('GO')].columns

        logit_preds, true_labels, pred_labels, tokenized_texts, val_f1_accuracy_list, \
            val_flat_accuracy_list, clf_reports = [], [], [], [], [], [], []

        # Predict
        for i, batch in enumerate(self.validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                outs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        for threshold in self.thresholds:
            pred_bools = [pl > threshold for pl in pred_labels]
            true_bools = [tl == 1 for tl in true_labels]
            val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
            val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100

            print('F1 Validation Accuracy: ', val_f1_accuracy)
            print('Flat Validation Accuracy: ', val_flat_accuracy)
            print('\n')
            val_f1_accuracy_list.append(val_f1_accuracy)
            val_flat_accuracy_list.append(val_flat_accuracy)
            clf_report = classification_report(true_bools, pred_bools, target_names=self.label_columns,
                                               output_dict=True)
            clf_reports.append(clf_report)
            print('THRESHOLD:', threshold)
            print(clf_report)

        df_best_threshold = pd.DataFrame({'Threshold': threshold, 'F1_Validation_Accuracy': val_f1_accuracy_list,
                                          'Flat_Validation_Accuracy': val_flat_accuracy_list}, index=[0])
        df_clf_reports = pd.DataFrame(clf_reports)
        df_clf_reports.to_csv(self.output_dir + 'clf_report.csv', sep=';', index=False)
        df_best_threshold.to_csv(self.output_dir + 'best_threshold.csv', sep=';', index=False)
        return df_clf_reports, df_best_threshold

    def save_model(self, output_dir):
        try:
            self.model.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")


def main():
    batch_size = 4
    epochs = 6
    num_labels = 230

    train_data = pd.read_csv('../data/ft_data.csv', sep=';')
    eval_data = pd.read_csv('../data/validate_data.csv', sep=';')
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    output_dir = '../output/'
    data_processor = DataProcessor(train_data[0:10], eval_data[0:10], tokenizer_name, batch_size=batch_size)
    train_labels, eval_labels, train_seq, eval_seq, label_columns = data_processor.define_labels()
    train_encodings, eval_encodings = data_processor.calculate_embeddings()

    # train_dataloader, validation_dataloader = DataProcessor.data_loader()

    train_inputs = torch.tensor(train_encodings.data['input_ids'])
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_encodings.data['attention_mask'])

    validation_inputs = torch.tensor(eval_encodings.data['input_ids'])
    validation_labels = torch.tensor(eval_labels)
    validation_masks = torch.tensor(eval_encodings.data['attention_mask'])

    train_data = TensorDataset(train_inputs, train_masks, train_labels, )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, )
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    thresholds = np.arange(0, 1, 0.1)

    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        problem_type="multi_label_classification",
        num_labels=230
    )

    model_trainer = ModelTrainer(model, num_labels, train_dataloader, validation_dataloader, tokenizer_name,
                                 eval_data, thresholds, epochs, batch_size, label_columns, output_dir)
    model_trainer.train()
    # model_trainer.save_model('C:/Users/sylwi/PycharmProjects/STSM/output/ft_protbert/')

    model_trainer.evaluate()


if __name__ == "__main__":
    main()
