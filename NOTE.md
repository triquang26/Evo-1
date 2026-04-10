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
