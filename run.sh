CUDA_VISIBLE_DEVICES=1 /opt/miniconda/envs/pytorch2.1/bin/python train_esm_finetune.py --is-peft False -pm esm2_t36_3B_UR50D --num-layers 0 --epochs 5

CUDA_VISIBLE_DEVICES=1 /opt/miniconda/envs/pytorch2.1/bin/python train_esm_finetune.py --is-peft True -pm esm2_t30_150M_UR50D -rmt 1  -nlora 30 -nr 2 -na 2 --epochs 5 -lr 3e-4


/opt/miniconda/envs/pytorch2.1/bin/python train_esm_mlp.py -pm esm2_t6_8M_UR50D -lr 1e-3 -hls "(64,)"

/opt/miniconda/envs/pytorch2.1/bin/python train_esm_mlp.py -pm esm2_t30_150M_UR50D -lr 3e-4 -hls "(128,)"

/opt/miniconda/envs/pytorch2.1/bin/python train_esm_mlp.py --pretrain-model "esm2_t33_650M_UR50D"  --learning-rate "1e-3" --hidden-layer-sizes "(128,)"




