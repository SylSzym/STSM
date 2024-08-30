import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer


class DataProcessor:

    def __init__(self, input_file):
        self.input_file = input_file

    def process_data(self):
        # Select label columns
        label_columns = self.input_file.loc[:, self.input_file.columns.str.startswith('GO')].columns.to_list()
        # Create a new DataFrame containing only the selected label columns
        df_labels = self.input_file[label_columns]
        # Convert the label columns to lists for each row
        labels_list = df_labels.values.astype(np.float32).tolist()

        # prepare sequences - replace rare aminoacid
        self.input_file['motif'] = self.input_file['motif'].apply(lambda x: x.replace("[UZOB]", "X"))

        input_seq = self.input_file['motif'].tolist()
        input_labels = labels_list

        return input_labels, input_seq, label_columns


class EmbeddingProcessor:
    def __init__(self, tokenizer_name, model, input_seq):
        self.tokenizer_name = tokenizer_name
        self.model = model
        self.input_seq = input_seq

    def get_embeddings(self):
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name, do_lower_case=False, max_length=512)
        embeddings = []
        for seq in self.input_seq:
            input = tokenizer(seq, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
            output = self.model(**input)
            embeddings.append(output['last_hidden_state'][0].tolist())
        df_embeddings = pd.DataFrame(embeddings)
        return df_embeddings


def main():
    input_file = pd.read_csv('../data/test_data.csv', sep=';')
    input_file = input_file[0:10]
    model_name = "Rostlab/prot_bert"

    model = BertModel.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=230
    )

    data_processor = DataProcessor(input_file)
    test_labels, test_seq, label_columns = data_processor.process_data()

    embedding_processor = EmbeddingProcessor(model_name, model, test_seq)
    embedding = embedding_processor.get_embeddings()
    uniprot_id = input_file['UniqueID']
    seq = pd.DataFrame(test_seq)
    seq.columns = ['motif']
    dataframe = pd.concat([uniprot_id, seq, embedding], axis=1)
    dataframe.to_csv('../output/protbert_test_embeddings.csv', sep=';', index=False)


if __name__ == "__main__":
    main()
