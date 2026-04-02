# Smoke test
```bash
python scripts/train.py \
  --dataset_config_path dataset/config.yaml \
  --max_samples_per_file 5 \    
  --batch_size 2 \              
  --max_steps 5 \               
  --num_workers 0 \             
  --disable_wandb \
  --horizon 16

```
## Check List
```bash
✅ Không có ImportError / ModuleNotFoundError
✅ Không có FileNotFoundError (tasks.jsonl, parquet, videos)
✅ Cache build thành công (thấy .pkl files được tạo)
✅ DataLoader tạo được batch
✅ Shapes đúng:
      images:  [B, max_views, 3, 448, 448]
      states:  [B, 24]
      actions: [B, horizon, 24]
✅ Forward pass không crash
✅ Loss in ra được (không NaN ngay từ đầu)
✅ 5 steps hoàn thành không lỗi
```
