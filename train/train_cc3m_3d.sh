source /home/aiops/wangzh/miniconda3/bin/activate
conda activate depth
export WANDB_API_KEY='bfd8edb90586095d3f73dd957cd98934dcdba95e'

EXP_NAME="3d-cc3m"
POSI="3d"
torchrun --nproc_per_node=8 train/train_retrival_cc3m.py --lr 4e-6 --weight_decay 1e-2 --log_scale 4.6052 --lora_rank -1 --common_pair 0.0 --para_gamma 0.05 --exp_name "$EXP_NAME" --warmup_length 5000 --amp  --pos_emb "$POSI" --resume
