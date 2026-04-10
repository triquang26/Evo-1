# Kế hoạch Tích hợp và Distillation: GR00T & GR00T-Mini

Mục tiêu mới ưu tiên tuyệt đối sự toàn vẹn của nền tảng (Platform Integrity) và độ chính xác của logic đánh giá (Evaluation Correctness). Phiên bản này áp đặt sinh viên (Student) phải mang DNA của GR00T gốc, kế thừa hoàn toàn cơ chế Lập trình hướng đối tượng (OOP), và xử lý triệt để Evaluation Benchmarking.

---

## 1. Thiết kế Hệ thống Sinh viên (GR00T-Mini Architecture)

Thay vì tích hợp Distillation ngoại lai (cross-architecture) vào Flow Matching, sinh viên sẽ là bản sao rỗng (Miniaturized Clone) thu được từ tập thiết kế `AutoConfig` của bản gốc `liorbenhorin-nv/groot-libero_10-64_40000`.

-   **Hành động**: Gọi `AutoConfig.from_pretrained()`, tinh cúp các layer `hidden_size` và `num_hidden_layers` sao cho phù hợp với tài nguyên VRAM.
-   **An toàn**: Cả Teacher và Student cùng chia sẻ chung một Hugging Face Processor, cùng cấu trúc ngõ vào (Input Modality) và cùng không gian Đầu ra (Action Space). Không xảy ra bất kỳ xung đột Data Alignment nào.

## 2. Bản Lề Kế thừa: Distillation Trainer Chuẩn OOP

Evo-1 sở hữu hệ cơ sở vô cùng mạnh là [Trainer](file:///mnt/data/sftp/data/quangpt3/Evo-1/src/evo/training/trainer.py#25-307) class tích hợp `Accelerator`.
Chúng ta sẽ viết file mới: `src/evo/training/groot_distill_trainer.py`.

*   `class Gr00tMiniDistillationTrainer(Trainer):` Kế thừa trực tiếp `src.evo.training.trainer.Trainer`.
*   **Overwrite [build_models](file:///mnt/data/sftp/data/quangpt3/Evo-1/src/evo/training/trainer.py#144-149):** Tự động tạo instance 2B Teacher và cấu hình instance Mini Student.
*   **Overwrite [compute_loss](file:///mnt/data/sftp/data/quangpt3/Evo-1/src/evo/training/trainer.py#170-199):** Gỡ bỏ Flow Matching. Thay thế bằng Pseudo-label Imitation (L1 Denoising/Chunking Loss) dựa trên Output đích của Teacher khi inference 1 batch data.
*   Mọi tính năng như Epoch tracking, Checkpointing, Multi-GPU Sync đều được chạy trơn tru do dùng chung engine.

## 3. Hệ chuẩn Đánh giá (Evaluation Protocol on LIBERO)

Theo như đã tìm kiếm tận cùng trong source code, Evaluation bằng bộ tools có sẵn của Evo-1 (viết tại [libero_client_4tasks.py](file:///mnt/data/sftp/data/quangpt3/Evo-1/evaluations/LIBERO/libero_client_4tasks.py)) chứa nhiều kịch bản Hardcode chuyên dụng cho mạng khác (đảo bits kẹp Gripper, zero-padding thiếu thông số). **Cố nhồi GR00T vào luồng evaluation này sẽ vỡ nát Correctness.**

Để **ĐẢM BẢO TÍNH CORRECTNESS 100%**:
1. Chúng ta từ bỏ client của Evo-1 đối với con GR00T-mini.
2. Chúng ta tái sử dụng toàn bộ quy trình Client-Server có trong `Isaac-GR00T`:
   * Server Policy ([gr00t/eval/run_gr00t_server.py](file:///mnt/data/sftp/data/quangpt3/Isaac-GR00T/gr00t/eval/run_gr00t_server.py)) sẽ load weights sau quá trình train. Cắm wrapper [Gr00tSimPolicyWrapper](file:///mnt/data/sftp/data/quangpt3/Isaac-GR00T/gr00t/policy/gr00t_policy.py#420-674) vào để chặn đứng và nắn chỉnh mọi bất đồng bộ shape.
   * Rollout Client ([gr00t/eval/rollout_policy.py](file:///mnt/data/sftp/data/quangpt3/Isaac-GR00T/gr00t/eval/rollout_policy.py)) truyền raw data cực mượt qua API.
Duy trì nguyên mô hình này, chúng ta được bảo lãnh bởi con số tỷ lệ thành công 98.45% Object Benchmark mà NVIDIA cam kết.

---

## User Review Required

1. **Gr00t-Mini Blueprint:** Bạn đã nắm được tư duy giảm Params trực tiếp trong `AutoConfig` thay vì viết file Python backbone mới từ đầu chưa?
2. **Loại bỏ client Evo-1 Libre:** Quyết định này là sinh tử để giữ Correctness. Bạn đồng ý ta sẽ dẹp module Eval Libre của Evo-1 sang một bên để dùng script của NVIDIA đi kèm trong `Isaac-GR00T` không? 

> [!NOTE]
> Mọi bước nghiên cứu, đào mã nguồn từ tận cùng các lớp Validation, Processor đều đã xong trên lý thuyết. Tôi không hề chạy (execute) bất kỳ file mã code nào! Khi Plan này được phê chuẩn, lệnh tiếp theo là xắn tay áo vào code.
