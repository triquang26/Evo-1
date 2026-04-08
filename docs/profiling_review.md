# Đánh giá hệ thống đo đạc (Profiling) trong `server.py`

Mình đã thực hiện phân tích dựa trên file code đo đạc (profiler) nằm trong `src/evo/serving/server.py`. Dưới đây là đánh giá chi tiết theo các tiêu chí bạn yêu cầu:

---

## 1. Đã "đầy đủ" (Tính toán FLOPs, Latency, Params) chưa?
**Đánh giá: CHƯA ĐẦY ĐỦ VÀ CÓ CHỖ CHƯA CHÍNH XÁC HOÀN TOÀN**

- **Parameter count (Số lượng tham số):** ❌ Code hiện tại hoàn toàn **không có phần tính toán parameters**.
- **FLOPs (Real FLOPs):** ⚠️ Đang dùng `torch.profiler` với cờ `with_flops=True`. 
  - *Điểm tốt:* Trả về được tổng lý thuyết (analytical FLOPs).
  - *Hạn chế:* PyTorch Profiler chỉ ước tính FLOPs dựa trên các phép toán ATen tiêu chuẩn. Nó không phải là "real hardware FLOPs" (tức là số FLOPs thực tế chạy trên GPU đo bằng phần cứng) và cũng có thể bỏ sót tính toán trong một số kernel custom. Với các mô hình lớn, việc dùng các thư viện chuyên dụng như `fvcore` hay `deepspeed` sẽ phân tích đúng các MACs trong Deep Learning hơn.
- **Latency & VRAM:** ✅ Được đo khá đầy đủ bằng `time.perf_counter()` và `torch.cuda.memory_allocated() / max_memory_allocated()`. Có sử dụng `torch.cuda.synchronize()` để đồng bộ GPU trước khi đo là thao tác cực kì chuẩn xác. Tuy nhiên, luồng đo latency lại thiếu thao tác "Warm-up" (giải thích ở phần 3).

## 2. OOP Oriented và Design Pattern đã phù hợp chưa?
**Đánh giá: CHƯA CHUẨN THIẾT KẾ (OOP POOR DESIGN)**

- **Global Variable (Biến toàn cục):** Việc khởi tạo `global_profiler = ProfilerState()` và sau đó dùng từ khoá `global global_profiler` trong hàm `infer_from_json_dict()` là vi phạm nguyên tắc Đóng gói (Encapsulation) của OOP, cản trở việc scale-up hoặc xử lý đa luồng (multi-threading/async operations).
- **Single Responsibility Principle (Vi phạm SRP):** Hàm `infer_from_json_dict` đang gánh vác quá nhiều trọng trách: Parse data -> Đẩy qua mô hình inference -> Bật tắt Profiler -> Tự lưu số liệu tính thời gian -> In log thống kê ra console. 
- **Design Pattern khuyên dùng:** Phần code đo đạc (latency, FLOPs, memory) nên được tách bạch hoàn toàn ra khỏi logic của Server. Chúng ta nên dùng **Decorator Pattern** (`@profile_inference`) hoặc **Context Manager Pattern** (`with Profiler(): ...`) để "bọc" quanh hàm chạy inference.

## 3. Cách sử dụng thế nào? Tính Correctness thế nào?
- **Cách đo thời gian (Correctness của Latency):** Có sử dụng `torch.cuda.synchronize()` ở hai đầu đoạn code là hành động **CHUẨN**. GPU hoạt động bất đồng bộ, nếu không có hàm này, `time.perf_counter()` sẽ chỉ đo thời gian CPU đẩy lệnh xuống GPU chứ không phải thời gian GPU chạy thực tế.
- **Vấn đề GPU Warm-up:** Quá trình tính latency bị trigger ngay từ API request đầu tiên. Thông thường, request đầu sẽ dính phí khởi tạo trễ (CUDA context creation, memory allocation). Đo Latency và đo FLOPs **BẮT BUỘC** phải có giai đoạn chạy thử/chạy mồi (Warm-up) vài batch rồi mới bắt đầu trigger Profiler thì dữ liệu mới chính xác.
- **Tạo ra File Trace (`inference_trace.json`) quá lạm dụng:** Nếu API bị call 100 lần, nó sẽ tự động chép đè file profiling trace 100 lần vào ổ cứng (chỉ khi enable biến môi trường). Đoạn code bật `ProfilerActivity` nên được kích hoạt thủ công ngoài model thay vì để trực tiếp vào middleware của web socket server.

---
**TÓM LẠI:** Metric tính được hiện tại mới chỉ mang tính chất tham khảo bề mặt. Code logic đang bị "hard-code" trộn lẫn vào nghiệp vụ phục vụ API. Bạn có muốn mình **tiến hành Refactor (viết lại) phần module đo đạc này** theo chuẩn OOP dưới dạng một `Context Manager` (vd: `with InferenceProfiler():`) và bổ sung luôn code đếm số lượng Parameters không?
