source /home/aiops/wangzh/miniconda3/bin/activate
conda activate depth


EXP_NAME="final-negative-large-wiseconv"
POSI="nodepth"
torchrun --nproc_per_node=1 train/test_large.py --lr 4e-6 --weight_decay 1e-2 --log_scale 4.6052 --lora_rank -1 --common_pair 0.0 --para_gamma 0.05 --exp_name "$EXP_NAME" --warmup_length 5000 --amp  --pos_emb "$POSI" #--resume
