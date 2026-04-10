import yaml
import subprocess
import time
import os
import sys
from pathlib import Path

# Thêm đường dẫn Libero để parse benchmark suite
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "evaluations" / "LIBERO" / "LIBERO"))

try:
    from libero.libero import benchmark
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False

class GrootEvaluator:
    def __init__(self, config_path: str):
        self.workspace = Path(__file__).resolve().parent.parent.parent.parent
        self.config_path = config_path
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f).get("evaluation", {})
            
        self.model_path = self.cfg.get("model_path", "")
        if not self.model_path:
            raise ValueError("model_path is required in the configuration file.")

        self.port = self.cfg.get("server_port", 5555)
        self.embodiment = self.cfg.get("embodiment_tag", "LIBERO_PANDA")
        self.n_episodes = self.cfg.get("n_episodes", 10)
        self.n_action_steps = self.cfg.get("n_action_steps", 8)

    def _get_env_names(self):
        """Lấy danh sách các environment để chạy đánh giá"""
        task_suite_name = self.cfg.get("task_suite", "")
        env_names_list = []
        
        if task_suite_name:
            if not LIBERO_AVAILABLE:
                print("Lỗi: Không import được thư viện libero. Kiểm tra ./evaluations/LIBERO/LIBERO")
                sys.exit(1)
            
            print(f"Đang phân tích Task Suite: {task_suite_name}")
            benchmark_dict = benchmark.get_benchmark_dict()
            if task_suite_name not in benchmark_dict:
                print(f"Lỗi: Task Suite {task_suite_name} không tồn tại trong libero benchmark.")
                sys.exit(1)
                
            task_suite = benchmark_dict[task_suite_name]()
            for i in range(task_suite.n_tasks):
                task = task_suite.get_task(i)
                env_names_list.append(task.env_name)
        else:
            # Fallback về một môi trường cụ thể nếu không khai báo suite
            env_name = self.cfg.get("env_name", "")
            if env_name:
                env_names_list.append(env_name)
            else:
                raise ValueError("Bạn phải truyền 'task_suite' hoặc 'env_name' trong config.")
                
        return env_names_list

    def run(self):
        os.chdir(self.workspace)
        env_names = self._get_env_names()
        
        print("==========================================")
        print(f"Bật Server GR00T (Port: {self.port})")
        print(f"Model: {self.model_path}")
        print("==========================================")
        
        server_cmd = [
            "python", "externals/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
            "--model-path", self.model_path,
            "--embodiment-tag", self.embodiment,
            "--use-sim-policy-wrapper",
            "--port", str(self.port)
        ]
        
        server_proc = subprocess.Popen(server_cmd)
        
        print("Đang đợi server khởi động (30s)...")
        time.sleep(30)
        
        try:
            for i, env_name in enumerate(env_names, 1):
                print(f"\n==========================================")
                print(f"[{i}/{len(env_names)}] Khởi chạy Client cho môi trường: {env_name}")
                print(f"==========================================")
                
                client_cmd = [
                    "python", "externals/Isaac-GR00T/gr00t/eval/rollout_policy.py",
                    "--n_episodes", str(self.n_episodes),
                    "--policy_client_port", str(self.port),
                    "--env_name", env_name,
                    "--n_action_steps", str(self.n_action_steps)
                ]
                
                subprocess.run(client_cmd)
        finally:
            print("\nEvaluation hoàn tất hoặc bị gián đoạn, đang đóng Server...")
            server_proc.terminate()
            server_proc.wait()
