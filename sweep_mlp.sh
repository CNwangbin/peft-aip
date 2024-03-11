#!/bin/bash

# YAML文件路径
yaml_file="sweep_mlp.yaml"

# 使用yq和jq提取Python程序名
program=$(yq -r '.program' < "$yaml_file")

# 构建基础Python命令
python_cmd="/opt/miniconda/envs/pytorch2.1/bin/python $program"

# 使用yq和jq提取pretrain-model参数，并添加到命令行
pretrain_models=$(yq -r '.parameters."pretrain-model".values[]' < "$yaml_file")
for model in $pretrain_models; do
  # 使用yq和jq提取learning-rate参数，并添加到命令行
  learning_rates=$(yq -r '.parameters."learning-rate".values[]' < "$yaml_file")
  for lr in $learning_rates; do
    # 使用yq和jq提取hidden-layer-sizes参数，并添加到命令行
    hidden_layer_sizes=$(yq -r '.parameters."hidden-layer-sizes".values[]' < "$yaml_file")
    for size in $hidden_layer_sizes; do
      # 构建完整命令行
      cmd="$python_cmd --pretrain-model \"$model\" --learning-rate \"$lr\" --hidden-layer-sizes \"$size\""
      # 打印或执行命令
    #   echo $cmd
      # 如果你想直接执行命令，请取消下一行的注释
      eval $cmd
    done
  done
done
