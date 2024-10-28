import pandas as pd
import numpy as np
import re
from transformers import T5Model, T5Tokenizer


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

        df_id_motif = pd.concat([pd.DataFrame(self.input_file['UniqueID']), pd.DataFrame(self.input_file['motif'])], axis = 1)
        df_id_motif.columns = ['UniqueID', 'motif']

        # prepare sequences - replace rare aminoacid
        self.input_file['motif'] = self.input_file['motif'].apply(lambda x: x.replace("[UZOB]", "X"))

        input_seq = self.input_file['motif'].tolist()

        return input_seq, df_id_motif, df_labels


class EmbeddingProcessor:
    def __init__(self, tokenizer_name, model, input_seq):
        self.tokenizer_name = tokenizer_name
        self.model = model
        self.input_seq = input_seq

    def get_embeddings(self):

        print('Calculating embeddings in progress ...')
        tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name, do_lower_case=False, max_length=512)
        embeddings = []
        for seq in self.input_seq:
            print(seq)
            input = tokenizer(seq, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
            output = self.model(**input, decoder_input_ids=self.model._shift_right(input['input_ids']))
            token_representations = output['last_hidden_state']
            # Skip tokens CLS and EOS (specjal tokens), calculate representative embbeddings for protein sequnece
            embedding = token_representations[0,1:len(seq)+1].mean(0)
            embeddings.append(embedding.detach().numpy())
        df_embeddings = pd.DataFrame(embeddings)
        return df_embeddings
        # embeddings = []
        # for seq in self.input_seq:
        #     inputs = tokenizer.batch_encode_plus(self.input_seq, padding="max_length", truncation=True, max_length=512,
        #                    return_tensors="pt")
        #     output = self.model(**inputs, decoder_input_ids=self.model._shift_right(inputs['input_ids']))
        #     embeddings.append(output['last_hidden_state'][0].tolist())
        # df_embeddings = pd.DataFrame(embeddings)
        # return df_embeddings

def main():
    input_file = pd.read_csv('data/test_data.csv', sep=';')
    model_name = "Rostlab/prot_t5_xl_uniref50"

    model = T5Model.from_pretrained(
        model_name
    )

    data_processor = DataProcessor(input_file)
    test_seq, df_id_motif, df_labels = data_processor.process_data()

    embedding_processor = EmbeddingProcessor(model_name, model, test_seq)
    embedding = embedding_processor.get_embeddings()
    dataframe = pd.concat([df_id_motif, embedding, df_labels], axis = 1)
    dataframe.to_csv('./output/prot_t5/prot_t5_test_embeddings.csv', sep=';', index=False)

if __name__ == "__main__":
    main()
