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

        # prepare sequences - replace rare aminoacid
        self.input_file['motif'] = self.input_file['motif'].apply(lambda x: " ".join(list(re.sub(r"[UZOB]", "X", x))))

        input_seq = self.input_file['motif'].tolist()
        input_labels = labels_list

        return input_labels, input_seq, label_columns


class EmbeddingProcessor:
    def __init__(self, tokenizer_name, model, input_seq):
        self.tokenizer_name = tokenizer_name
        self.model = model
        self.input_seq = input_seq

    def get_embeddings(self):

        print('Calculating embeddings in progress ...')
        tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name, do_lower_case=False, max_length=512)
        inputs = tokenizer.batch_encode_plus(self.input_seq, padding="max_length", truncation=True, max_length=512,
                           return_tensors="pt")
        # decoder_input_ids = self.model._shift_right(inputs['input_ids'])
        outputs = self.model(**inputs, decoder_input_ids=None)
        print(outputs)
        embeddings = pd.DataFrame(outputs['last_hidden_state'][0].tolist())
        return embeddings

    # with torch.no_grad():
    #     # Forward pass
    #     outs = self.model(input_ids=b_input_ids, attention_mask=b_input_mask, decoder_input_ids=decoder_input_ids)
    #     b_logit_pred = outs[0]
    #     protein_logits_pred = []
    #     for i in range(len(b_logit_pred)):
    #         protein_logits_pred.append(b_logit_pred[i].mean(dim=0))
    #         protein_logits_pred = torch.tensor(protein_logits_pred)


def main():
    input_file = pd.read_csv('../data/test_data.csv', sep=';')
    input_file = input_file[0:10]
    model_name = "Rostlab/prot_t5_xl_uniref50"

    model = T5Model.from_pretrained(
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
    dataframe.to_csv('../output/prot_t5_embeddings.csv', sep=';', index=False)


if __name__ == "__main__":
    main()