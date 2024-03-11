import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import esm
import re
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

from dataset import ESM2_MODEL_LIST, DEFAULT_ESM_MODEL


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        
        # 添加隐藏层
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
        
        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # 最后一层不加激活函数
        return x


class FineTuneESM(nn.Module):
    def __init__(self, 
                 num_layers, 
                 pretrained_model_name=DEFAULT_ESM_MODEL, 
                 remove_top_layers=0, 
                 hidden_layer_sizes=(64,)):
        super(FineTuneESM, self).__init__()
        if pretrained_model_name not in ESM2_MODEL_LIST:
            print(
                f"Model dir '{pretrained_model_name}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            pretrained_model_name = DEFAULT_ESM_MODEL
        self.repr_layer = int(re.findall(r't(\d+)', pretrained_model_name)[0])-remove_top_layers
        print("using repr layer", self.repr_layer)
        self.esm_model, _ = esm.pretrained.load_model_and_alphabet(pretrained_model_name)
        self.num_esm_layers = len(self.esm_model.layers)
        self.finetune_layers = self.esm_model.layers[self.num_esm_layers-num_layers:self.num_esm_layers]
        for p in self.esm_model.parameters():
            p.requires_grad = False

        for layer in self.finetune_layers:
            for param in layer.parameters():
                param.requires_grad = True
        self.classifier = MLPClassifier(self.esm_model.embed_dim, 1, hidden_layer_sizes)
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
    def __init__(self, 
                 num_end_lora_layers, 
                 num_lora_r, 
                 num_lora_alpha, 
                 inference_mode=False, 
                 lora_dropout=0.1, 
                 bias="none", 
                 pretrained_model_name=DEFAULT_ESM_MODEL, 
                 remove_top_layers=0):
        super(PeftESM, self).__init__()
        if pretrained_model_name not in ESM2_MODEL_LIST:
            print(
                f"Model dir '{pretrained_model_name}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            pretrained_model_name = DEFAULT_ESM_MODEL
        self.repr_layer = int(re.findall(r't(\d+)', pretrained_model_name)[0]) - remove_top_layers
        print("using repr layer", self.repr_layer)
        self.esm_model, _ = esm.pretrained.load_model_and_alphabet(pretrained_model_name)
        self.num_esm_layers = len(self.esm_model.layers)
        for p in self.esm_model.parameters():
            p.requires_grad = False
        self.target_model_dict = {"qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"]}
        self.esm_model = self.lora_model(self.esm_model, num_end_lora_layers, num_lora_r, num_lora_alpha, inference_mode=inference_mode, lora_dropout=lora_dropout, bias=bias)
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
    
    def lora_model(self, model, num_end_lora_layers, num_lora_r, num_lora_alpha, inference_mode=False, lora_dropout=0.1, bias="none"):
        target_modules = []
        start_layer_idx = self.num_esm_layers - num_end_lora_layers
        for idx in range(start_layer_idx, self.num_esm_layers):
            for layer_name in self.target_model_dict["qkvo"]:
                target_modules.append(f"layers.{idx}.self_attn.{layer_name}")

        peft_config = LoraConfig(inference_mode=inference_mode,
                                r=num_lora_r,
                                lora_alpha=num_lora_alpha,
                                target_modules=target_modules,
                                lora_dropout=lora_dropout,
                                bias=bias)
        model = get_peft_model(model, peft_config)
        return model
    


if __name__ == '__main__':
    import pandas as pd
    from dataset import EsmDataset
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = FineTuneESM(1, 'esm2_t33_650M_UR50D')
    train_df = pd.read_pickle('/home/wangbin/peft-aip/data/data_AIP-MDL/train_df.pkl')
    train_dataset = EsmDataset(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn)
    batch_data = next(iter(train_loader))
    batch_data = {k: v.cuda() for k, v in batch_data.items()}
    model = model.cuda()
    output = model(**batch_data)

    # model = PeftESM(1, 2, 0.1, 'esm2_t36_3B_UR50D')
    # output = model(**batch_data)