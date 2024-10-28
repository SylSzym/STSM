import pandas as pd
import numpy as np
import esm
import torch


class DataProcessor:

    def __init__(self, input_file):
        self.input_file = input_file

    def process_data(self):
        # Select label columns
        label_columns = self.input_file.loc[:, self.input_file.columns.str.startswith('GO')].columns.to_list()
        # Create a new DataFrame containing only the selected label columns
        df_labels = self.input_file[label_columns]

        list_of_sequences = []
        #create list of sequences
        for i in range(len(self.input_file)):
            list_of_sequences.append([self.input_file['UniqueID'].iloc[i], self.input_file['motif'].iloc[i]])

        return df_labels, list_of_sequences


class EmbeddingProcessor:
    def __init__(self, model, input):
        # self.tokenizer_name = tokenizer_name
        self.model = model
        self.input = input

    def get_embeddings(self):
        model, alphabet = self.model
        batch_converter = alphabet.get_batch_converter()
        batch_ids, batch_sequences, batch_tokens = batch_converter(self.input)
        df_id_motif = pd.concat([pd.DataFrame(batch_ids), pd.DataFrame(batch_sequences)], axis = 1)
        df_id_motif.columns = ['UniqueID', 'motif']

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

        embeddings = []
        for i, (seq) in enumerate(self.input):
        # Skip tokens CLS and EOS (specjal tokens)
            embedding = token_representations[i, 1:len(seq)+1].mean(0)
            embeddings.append(embedding.detach().numpy())

            dataframe = pd.concat([df_id_motif, pd.DataFrame(embeddings)], axis = 1)
        
        return dataframe


def main():
    input_file = pd.read_csv('data/test_data.csv', sep=';')
    input_file = input_file[0:10]

    model = esm.pretrained.esm2_t33_650M_UR50D()

    data_processor = DataProcessor(input_file)
    df_labels, list_of_sequences = data_processor.process_data()

    embedding_processor = EmbeddingProcessor(model, list_of_sequences)
    df_embeddings = embedding_processor.get_embeddings()
    dataframe = pd.concat([df_embeddings, df_labels], axis = 1)
    dataframe.to_csv('output/esm2_test_embeddings.csv', sep=';', index=False)
    
if __name__ == "__main__":
    main()