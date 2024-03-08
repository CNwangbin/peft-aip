import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import esm
import re

from dataset import ESM2_MODEL_LIST, DEFAULT_ESM_MODEL

class FineTuneESM(nn.Module):
    def __init__(self, num_layers, pretrained_model_name=DEFAULT_ESM_MODEL):
        super(FineTuneESM, self).__init__()
        if pretrained_model_name not in ESM2_MODEL_LIST:
            print(
                f"Model dir '{pretrained_model_name}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            pretrained_model_name = DEFAULT_ESM_MODEL
        self.repr_layer = int(re.findall(r't(\d+)', pretrained_model_name)[0])
        print("using repr layer", self.repr_layer)
        self.esm_model, _ = esm.pretrained.load_model_and_alphabet(pretrained_model_name)
        self.finetune_layers = self.esm_model.layers[-num_layers:]
        for layer in self.esm_model.layers:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.finetune_layers:
            for param in layer.parameters():
                param.requires_grad = True
        self.classifier = nn.Linear(self.esm_model.embed_dim, 1)
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, input_ids, input_mask, labels=None):
        embedding = self.esm_model(input_ids, repr_layers=[self.repr_layer])['representations'][self.repr_layer]
        embedding_sum  = embedding * input_mask.unsqueeze(-1)
        embedding = embedding_sum.sum(
            dim=1, keepdim=False) / input_mask.sum(1).unsqueeze(-1)
        logits = self.classifier(embedding)
        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits.view(-1), labels.view(-1))
        return loss, logits

class PeftESM(nn.Module):
    def __init__(self, num_layers, pretrained_model_name=DEFAULT_ESM_MODEL):
        super(PeftESM, self).__init__()
        if pretrained_model_name not in ESM2_MODEL_LIST:
            print(
                f"Model dir '{pretrained_model_name}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            pretrained_model_name = DEFAULT_ESM_MODEL
        self.repr_layer = int(re.findall(r't(\d+)', pretrained_model_name)[0])
        print("using repr layer", self.repr_layer)
        self.esm_model, _ = esm.pretrained.load_model_and_alphabet(pretrained_model_name)
        self.finetune_layers = self.esm_model.layers[-num_layers:]
        for layer in self.esm_model.layers:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.finetune_layers:
            for param in layer.parameters():
                param.requires_grad = True
        self.classifier = nn.Linear(self.esm_model.embed_dim, 1)
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, input_ids, labels=None):
        embedding = self.esm_model(input_ids, repr_layers=[self.repr_layer])['representations'][self.repr_layer]
        embedding = torch.mean(embedding, axis=1)
        logits = self.classifier(embedding)
        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits.view(-1), labels.view(-1))
        return loss, logits


if __name__ == '__main__':
    import pandas as pd
    from dataset import EsmDataset
    model = FineTuneESM(1, 'esm2_t36_3B_UR50D')
    train_df = pd.read_pickle('/home/wangbin/peft-aip/data/data_AIP-MDL/train_df.pkl')
    train_dataset = EsmDataset(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn)
    batch_data = next(iter(train_loader))
    # batch_data = {k: v.float() for k, v in batch_data.items()}
    output = model(**batch_data)

    # target_model_dict = {
    #     "qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    # }
    # from peft import LoraConfig, get_peft_model
    # peft_config = LoraConfig(task_type="SEQ_CLS",
    #     inference_mode=False,
    #     target_modules=target_model_dict['qkvo'],
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.1)

    # # 加⼊PEFT策略
    # peft_model = get_peft_model(model, peft_config)