# Evo-1: VLA Models Framework

This repository supports continuous distillation and evaluation of Foundation Vision Language Action (VLA) models, such as GR00T and GR00T-Mini.

## Evaluation

To evaluate GR00T (Teacher) or GR00T-Mini (Student) using the official Isaac-GR00T Rollout server-client architecture, simply run the Python evaluation script:

```bash
python scripts/evaluate_groot.py
```

### Configuration

By default, the script reads all evaluation parameters from `configs/eval/groot.yaml`. You don't need to pass command line arguments. 

Open `configs/eval/groot.yaml` to configure:
- `model_path`: Path to your checkpoint.
  - *VD Gốc (Teacher):* `"liorbenhorin-nv/groot-libero_10-64_40000"`
  - *VD Distill (Student):* `"outputs/checkpoints/stage1_v02/step_final"`
- `env_name`: Tên môi trường cần test.
- `n_episodes`: Số lượng episodes.

*Optional:* If you want to use a different config file, you can pass it via `--config`:
```bash
python scripts/evaluate_groot.py --config configs/eval/my_custom_eval.yaml
```
