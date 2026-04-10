tmux evo1_benchmark_4step -> evo1 server, port 9010
tmux evo1_benchmark_4steo_client ->evo1 student


SmolVLA evaluation code:
lerobot-eval \
  --policy.path="HuggingFaceVLA/smolvla_libero" \
  --env.type=libero \
  --env.task=libero_spatial,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --env.max_parallel_tasks=1




  python scripts/distillation/transfer_weights_groot.py \
    --config configs/train/distill_groot.yaml \
    --output_ckpt root_mini_checkpoint.pt

traine code
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --dynamo_backend inductor \
    src/evo/training/snapflow_distill_trainer.py configs/train/snapflow_libero.yaml
dataset download code:
# Tải dataset libero_10
huggingface-cli download --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --local-dir /mnt/data/sftp/data/quangpt3/Evo-1/src/evo/data/libero_10 --local-dir-use-symlinks False

# Tải dataset libero_spatial
huggingface-cli download --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --local-dir /mnt/data/sftp/data/quangpt3/Evo-1/src/evo/data/libero_spatial --local-dir-use-symlinks False

# Tải dataset libero_object
huggingface-cli download --repo-type dataset IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot --local-dir /mnt/data/sftp/data/quangpt3/Evo-1/src/evo/data/libero_object --local-dir-use-symlinks False


same for goal
