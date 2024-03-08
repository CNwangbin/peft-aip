import gc
from typing import Dict
import esm
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


ESM2_MODEL_LIST = [
    'esm2_t6_8M_UR50D',
    'esm2_t12_35M_UR50D',
    'esm2_t30_150M_UR50D',
    'esm2_t33_650M_UR50D',
    'esm2_t36_3B_UR50D',
    'esm2_t48_15B_UR50D',
]
DEFAULT_ESM_MODEL = 'esm2_t33_650M_UR50D'

class EsmDataset(Dataset):
    """ESMDataset."""

    def __init__(self,
                 df: pd.DataFrame = None):
        super().__init__()
        if df is not None:
            self.df = df
        else:
            print('Please use the construct_df method to generate the df dataset')


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_token = self.df.tokens[idx]
        label = self.df.labels[idx]
        return seq_token, label


    def collate_fn(self, examples) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model
        used."""
        seq_token_list = [ex[0] for ex in examples]
        label_list = [ex[1] for ex in examples]

        # alphabet.padding_idx = 1
        protein_tensors = [torch.tensor(t) for t in seq_token_list]

        # pad_sequence for token padding
        padded_protein_tensors = pad_sequence(protein_tensors, batch_first=True, padding_value=1)
        input_mask = torch.ones(padded_protein_tensors.size(0), padded_protein_tensors.size(1), dtype=torch.long)
        input_mask[padded_protein_tensors == 1] = 0 # mask padding tokens
        input_mask[padded_protein_tensors == 0] = 0 # mask cls tokens
        input_mask[padded_protein_tensors == 2] = 0 # mask eos tokens

        encoded_inputs = {
            'input_ids': padded_protein_tensors,
            'labels': torch.tensor(label_list),
            'input_mask': input_mask.float()
        }

        return encoded_inputs
    
    def construct_df(self, 
                     model_dir: str = 'esm2_t33_650M_UR50D', 
                     pos_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/training-pos.txt', 
                     neg_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/training-neg.txt'):
        if model_dir not in ESM2_MODEL_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        _, alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.batch_converter = alphabet.get_batch_converter()

        _, pos_seqs = read_fasta(pos_file_path)
        _, neg_seqs = read_fasta(neg_file_path)
        df = self.get_pos_neg_df(pos_seqs, neg_seqs)
        return df

    def get_pos_neg_df(self, pos_seqs, neg_seqs):
        pos_tokens = self.get_seq_tokens(pos_seqs)
        neg_tokens = self.get_seq_tokens(neg_seqs)
        pos_df = pd.DataFrame({'sequences': pos_seqs, 'labels': [1] * len(pos_seqs), 'tokens': pos_tokens})
        neg_df = pd.DataFrame({'sequences': neg_seqs, 'labels': [0] * len(neg_seqs), 'tokens': neg_tokens})
        return pd.concat([pos_df, neg_df], ignore_index=True)

    def get_seq_tokens(self, seqs):
        tokens_list = list()
        for i in range(len(seqs)):
            _, _, seq_tokens = self.batch_converter([('', seqs[i])])
            tokens_list.append(seq_tokens[0].numpy().tolist())
        return tokens_list

    def free_memory(self, esm_model):
        del esm_model
        gc.collect()
        print('Delete the esm model, free memory!')


def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs


if __name__ == '__main__':
    
    ## generate the df dataset

    # esm_dataset = EsmDataset()
    # train_df = esm_dataset.construct_df(model_dir = 'esm2_t33_650M_UR50D',
    #                                     pos_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/training-pos.txt', 
    #                                     neg_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/training-neg.txt')
    # valid_df = esm_dataset.construct_df(model_dir = 'esm2_t33_650M_UR50D',
    #                                     pos_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/validation-pos.txt',
    #                                     neg_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/validation-neg.txt')
    # test_df = esm_dataset.construct_df(model_dir = 'esm2_t33_650M_UR50D',
    #                                    pos_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/test-pos.txt',
    #                                     neg_file_path='/home/wangbin/peft-aip/data/data_AIP-MDL/test-neg.txt')
    # train_df.to_pickle('/home/wangbin/peft-aip/data/data_AIP-MDL/train_df.pkl')
    # valid_df.to_pickle('/home/wangbin/peft-aip/data/data_AIP-MDL/valid_df.pkl')
    # test_df.to_pickle('/home/wangbin/peft-aip/data/data_AIP-MDL/test_df.pkl')

    train_df = pd.read_pickle('/home/wangbin/peft-aip/data/data_AIP-MDL/train_df.pkl')
    train_dataset = EsmDataset(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn)
    batch_data = next(iter(train_loader))
    model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    output = model(batch_data['input_ids'], repr_layers=[33])['representations'][33]