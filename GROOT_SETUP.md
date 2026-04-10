# Hướng dẫn Evaluate GR00T (1 Episode / 1 Task)

Dưới đây là một bản hướng dẫn chi tiết để bạn có thể **tự chạy thủ công** test thử 1 task - 1 episode qua framework Evo-1.
Hệ thống Evo-1 dự án của bạn sử dụng phiên bản **Isaac-GR00T** tích hợp sẵn bằng Github Submodule (nằm ở `externals/Isaac-GR00T`), và bạn đã có sẵn luôn kịch bản tự động hóa là file `scripts/evaluate_groot.py`.

## 1. Môi trường (Conda / UV)
Isaac-GR00T được NVIDIA thiết kế để dùng chung với **uv** (trình quản lý package của Rust) để cài phụ thuộc và chống treo, tuy nhiên dùng kèm với `conda` để quản lý phiên bản Python là an toàn nhất. Bạn dùng **Python 3.10**:

**Cách setup môi trường**:
1. Mở terminal, truy cập vào root của project `Evo-1`.
2. Khởi tạo một môi trường mới:
```bash
conda create -n groot_test python=3.10 -y
conda activate groot_test
```
3. Cài đặt `uv` và trỏ vào dự án Isaac-GR00T để cài dependency:
```bash
pip install uv

# Tạt qua thu mục Isaac-GR00T tải dependencies
cd externals/Isaac-GR00T
uv pip install -e .

# Cài đặt tiếp các module ở chính Evo-1 (để script ở gốc tìm được module nội bộ)
cd ../..
pip install -e .
```

*Lưu ý: Nếu không muốn tự dùng Conda, bạn có thể chạy file cài tự động bằng shell script: `bash externals/Isaac-GR00T/scripts/deployment/dgpu/install_deps.sh`. Nó sẽ tự sinh ra `.venv`.*

## 2. Cấu hình để chạy đúng 1 Task, 1 Episode
Mở file `configs/eval/groot.yaml`, bạn sẽ thấy cấu hình mặc định là đang phân tập toàn bộ suite `libero_10` và test nhiều lần. Để giới hạn phạm vi rút gọn thành 1 episode duy nhất cho 1 task, bạn hãy sửa lại toàn bộ nội dung file đó thành như sau:

```yaml
evaluation:
  model_path: "liorbenhorin-nv/groot-libero_10-64_40000"
  
  # Bỏ chuỗi này thành rỗng ("") để script không scan qua cả cụm task (toàn bộ 10 tasks)
  task_suite: ""
  
  # Chỉ định 1 task vụ cụ thể
  env_name: "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_soup_in_the_basket"
  
  embodiment_tag: "LIBERO_PANDA"
  server_port: 5555
  
  # Sửa xuống còn 1 episode
  n_episodes: 1
  n_action_steps: 8
```

## 3. Lệnh Evaluation
Bạn không cần tự tách Terminal ra chạy Server 1 màn hình và Client 1 màn hình đâu vì file `scripts/evaluate_groot.py` mà bạn có trong mã nguồn đã đảm nhận việc đó. File này gọi ngầm Server, đợi 30 giây rồi phóng Client vào, cuối cùng xoá port tự động.

Chỉ cần gõ **một dòng duy nhất** này từ thư mục gốc của project `Evo-1` (khi đã activate môi trường):

```bash
python scripts/evaluate_groot.py --config configs/eval/groot.yaml
```

**Workflow chạy sẽ như thế này để bạn dễ theo dõi:**
1. Nó sẽ báo bật Server ở cấu hình `port=5555` và bắt đầu tải model `liorbenhorin-nv/groot-libero_10-64_40000` (nặng 3 tỷ tham số) từ HuggingFace vào VRAM.
2. Tại đây sẽ có delay khoảng 30s để server load weights xong.
3. Sau 30s, file Rollout Policy Client của NVIDIA (`externals/Isaac-GR00T/gr00t/eval/rollout_policy.py`) được kích hoạt, tự động gọi vào giả lập Libero. Mô phỏng bắt đầu 1 episode của task.
4. Chạy xong, Server tự động bị ngắt (terminate) để giải phóng RAM GPU một cách gọn gàng.
