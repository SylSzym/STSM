import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, hamming_loss

class Preprocess_data:

    def __init__(self, input_file):
        self.input_file = input_file
    def get_labels(self):
        # Select label columns
        label_columns = self.input_file.loc[:, self.input_file.columns.str.startswith('GO')].columns.to_list()
        # Create a new DataFrame containing only the selected label columns
        df_labels_train = self.input_file[label_columns]
        # Convert the label columns to lists for each row
        labels_list_train = df_labels_train.values.astype(np.float32).tolist()
        labels = labels_list_train
        return labels, label_columns


class Classisfication_approaches:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def one_vs_rest(self):
        classifier = OneVsRestClassifier(LogisticRegression(class_weight='balanced'), n_jobs=-1)
        classifier.fit(self.x_train, self.y_train)
        pred_labels = classifier.predict(self.x_test)

        return pred_labels

    def classifier_chain(self):
        classifier = ClassifierChain(LogisticRegression(C=1))
        classifier.fit(self.x_train, self.y_train)
        pred_labels = classifier.predict(self.x_test)

        return pred_labels

    def nn_classifier(self):
        pass

class Report:
    def __init__(self, x_train, x_test, y_train, y_test, y_pred, thresholds, label_columns, output_dir):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.thresholds = thresholds
        self.label_columns = label_columns
        self.output_dir = output_dir
    def get_classification_report(self):
        true_labels, pred_labels, val_f1_accuracy_list, val_flat_accuracy_list, \
            threshold_list, hamming_loss_list, clf_reports = [], [], [], [], [], [], []
        # Flatten outputs
        true_labels = [item for sublist in self.y_test for item in sublist]
        pred_labels = [item for sublist in self.y_pred for item in sublist]

        # Calculate Accuracy
        for threshold in thresholds:
            pred_bools = [pl > threshold for pl in pred_labels]
            true_bools = [tl == 1 for tl in true_labels]
            val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
            val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100
            hamming_loss_value = round(hamming_loss(true_bools, pred_bools), 2)

            '''Hamming Loss is the fraction of incorrectly predicted class labels to the total number of actual labels.
            Macro averaged F1 score is good metric for multilabel classification tasks however doesn’t work well when 
            it comes to highly imbalanced distribution of tags.'''


            print('F1 Validation Accuracy: ', val_f1_accuracy)
            print('Flat Validation Accuracy: ', val_flat_accuracy)
            print('Hamming Loss: ', hamming_loss)
            print('\n')
            val_f1_accuracy_list.append(val_f1_accuracy)
            val_flat_accuracy_list.append(val_flat_accuracy)
            hamming_loss_list.append(hamming_loss_value)
            threshold_list.append(threshold)
            clf_report = classification_report(true_bools, pred_bools, target_names=self.label_columns,
                                               output_dict=True)
            clf_reports.append(clf_report)
            print('THRESHOLD:', threshold)
            print(clf_report)

        df_best_threshold = pd.DataFrame({'Threshold': threshold_list, 'F1_Validation_Accuracy': val_f1_accuracy_list,
                                          'Flat_Validation_Accuracy': val_flat_accuracy_list,
                                          'Hamming_Loss': hamming_loss_list})
        df_clf_reports = pd.DataFrame(clf_reports)
        df_clf_reports.to_csv(self.output_dir + 'ft_prot_t5_clf_report.csv', sep=';', index=False)
        df_best_threshold.to_csv(self.output_dir + 'ft_prot_t5_best_threshold.csv', sep=';', index=False)
        return df_clf_reports, df_best_threshold


def main():

    raw_data = pd.read_csv('../data/ft_data.csv')
    thresholds = np.arange(0, 1, 0.1)
    output_dir = '../output/'
    labels, label_columns = Preprocess_data.get_labels(raw_data)
    # data_processor = Classisfication_approaches(x_train, x_test, y_train, y_test)
    classifier = Classisfication_approaches.classifier_chain()

if __name__ == "__main__":
    main()
