# Experiment 2: Shallow-π Distillation

Tài liệu này mô tả ngắn gọn về thiết lập và các thông số cho phương pháp Distillation đang được sử dụng (Shallow-π) cho mô hình Evo-1, được gọi là **Experiment 2**.

## 1. Phương pháp hiện tại (Approach)
Thay vì sử dụng các phương pháp Feature Matching truyền thống, Experiment 2 áp dụng framework **Shallow-π** để khắc phục hiện tượng "gradient shock" và over-constraining:
- **Uniform Subsampling (Lấy mẫu đồng đều)**: Thay vì chọn các layer ở đầu hoặc cuối, Student model được khởi tạo bằng cách lấy các layer cách đều nhau từ mô hình Teacher. Điều này giúp giữ nguyên phân phối đặc trưng giữa các tầng.
- **Cross-Attention Distillation (KL Divergence)**: Việc transfer kiến thức (knowledge distillation) không thực hiện qua MSE loss trên output của VLM, mà tập trung vào **Cross-Attention** trong Action Head (Action-to-Vision/Language) thông qua KL Divergence ($L_{kd\_attn}$).
- **Middle-Layer Synchronization**: Thay vì đồng bộ tất cả các layer, quá trình distillation được áp dụng ở **layer giữa** của module transformer.

## 2. Thông số & Cấu hình (Configurations)
File config chính đang được sử dụng: `configs/train/distill.yaml`.

### 2.1 Cấu trúc Model (Student vs Teacher)
| Thành phần | Teacher (Evo-1) | Student (Experiment 2) |
| :--- | :--- | :--- |
| **VLM Layers** | 14 layers | **4 layers** |
| **Action Head Layers**| 8 layers | **4 layers** |
| **Hidden / Embed Dim**| 1024 / 896 | 1024 / 896 (Giữ nguyên để tránh gradient shock) |

### 2.2 Hyperparameters & Loss
- **Loss Weights**:
  - Task Loss ($\lambda_{task}$): `0.5`
  - Velocity KD Loss ($\lambda_{kd\_vel}$): `1.0`
  - Cross-Attention KD Loss ($\lambda_{kd\_attn}$): `1.0`
- **Attention KD Mapping**:
  - Student Layer: `1` (Layer thứ 2)
  - Teacher Layer: `3` (Layer thứ 4)
- **Training Config**: 
  - `lr`: `2e-4`
  - Cập nhật trọng số của `vlm` và `action_head` đều được bật (`finetune_vlm: True`).

## 3. Khởi tạo Trọng số (Initial Weights)
Trước khi quá trình distillation bắt đầu, script `transfer_weights_shallow_pi.py` được sử dụng để mapping parameters trực tiếp từ Teacher sang Student:
- **VLM Mapping**: Teacher `[0, 4, 9, 13]` $\rightarrow$ Student `[0, 1, 2, 3]`
- **Action Head Mapping**: Teacher `[0, 2, 4, 6]` $\rightarrow$ Student `[0, 1, 2, 3]`

Việc ánh xạ này cung cấp cho Student model một xuất phát điểm cực tốt (initial state gần với không gian đặc trưng của pre-trained Teacher), rút ngắn đáng kể thời gian hội tụ.
