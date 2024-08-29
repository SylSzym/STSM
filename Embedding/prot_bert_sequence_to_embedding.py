import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer


class DataProcessor:

    def __init__(self, data_test):
        self.data_test = data_test

    def process_data(self):
        not_chosen_columns = self.data_test.loc[:, ~self.data_test.columns.str.startswith('GO')]
        # Select label columns
        label_columns = [col for col in self.data_test.columns if col not in not_chosen_columns]
        # Create a new DataFrame containing only the selected label columns
        df_labels_test = self.data_test[label_columns]
        # Convert the label columns to lists for each row
        labels_list_test = df_labels_test.values.astype(np.float32).tolist()

        # prepare sequences - replace rare aminoacid
        self.data_test['motif'] = self.data_test['motif'].apply(lambda x: x.replace("[UZOB]", "X"))

        test_seq = self.data_test['motif'].tolist()
        test_labels = labels_list_test

        return test_labels, test_seq, label_columns


class EmbeddingProcessor:
    def __init__(self, tokenizer_name, model, test_seq):
        self.tokenizer_name = tokenizer_name
        self.model = model
        self.test_seq = test_seq

    def get_embeddings(self):

        print('Calculating embeddings in progress ...')
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name, do_lower_case=False, max_length=512)
        inputs = tokenizer(self.test_seq, padding="max_length", truncation=True, max_length=512,
                           return_tensors="pt")
        print('Tokenization done')
        outputs = self.model(**inputs)
        embeddings = pd.DataFrame(outputs['last_hidden_state'][0].tolist())
        return embeddings
def main():
    data_test = pd.read_csv('../data/test_data.csv', sep =';')
    data_test = data_test[0:10]

    model = BertModel.from_pretrained(
        "Rostlab/prot_bert",
        problem_type="multi_label_classification",
        num_labels=230
    )

    tokenizer_name = "Rostlab/prot_bert"

    data_processor = DataProcessor(data_test)
    test_labels, test_seq, label_columns = data_processor.process_data()

    embedding_processor = EmbeddingProcessor(tokenizer_name, model, test_seq)
    embedding = embedding_processor.get_embeddings()
    df_test_seq = pd.DataFrame(test_seq)
    df_test_seq.columns = ['motif']
    dataframe = pd.concat([df_test_seq, embedding], axis=1)
    dataframe.to_csv('./output/protbert_embeddings.csv', sep=';', index=False)

if __name__ == "__main__":
    main()
