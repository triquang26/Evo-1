import time
import torch
import numpy as np
import os

class InferenceProfiler:
    """
    Context manager for properly profiling PyTorch inference performance.
    Handles device synchronization, warmup skips, latency tracking, VRAM, and optional PyTorch Profiler tracing.
    """
    def __init__(self, warmup_steps=3, enable_flops_profiling=False, device="cuda"):
        self.warmup_steps = warmup_steps
        self.enable_flops_profiling = enable_flops_profiling
        self.device = device
        self.count = 0
        self.latencies = []
        self.peak_vrams = []
        self.curr_vrams = []
        self._start_time = None
        self.prof = None

    def __enter__(self):
        self.count += 1
        # Skip profiling during warmup steps
        if self.count <= self.warmup_steps:
            print(f"[Profiler] Skipping step {self.count} (Warmup)")
            return self

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        
        if self.enable_flops_profiling:
            from torch.profiler import profile, ProfilerActivity
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                                record_shapes=True, profile_memory=True, with_flops=True)
            self.prof.__enter__()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exclude from stats if an exception occurred
        if exc_type is not None:
            if self.prof is not None:
                self.prof.__exit__(exc_type, exc_val, exc_tb)
            return False

        if self.count <= self.warmup_steps:
            return False

        torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - self._start_time
        
        if self.enable_flops_profiling and self.prof is not None:
            self.prof.__exit__(exc_type, exc_val, exc_tb)
            total_flops = sum([getattr(evt, 'flops', 0) for evt in self.prof.key_averages()])
            print(f"\n============ [COMPUTE METRICS] ============")
            print(f"Total Evaluated Compute : {total_flops / 1e12:.4f} TFLOPs")
            print(f"===========================================\n")
            # print(self.prof.key_averages().table(sort_by="flops", row_limit=15))
            self.prof.export_chrome_trace("inference_trace.json")
            print(f"[Profiling] Detailed Chrome trace saved to inference_trace.json")
            self.prof = None
            
        peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        current_vram = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        
        self.latencies.append(latency)
        self.peak_vrams.append(peak_vram)
        self.curr_vrams.append(current_vram)
        
        self.print_current_stats(latency, peak_vram, current_vram)
        return False

    def print_current_stats(self, latency, peak_vram, current_vram):
        stats = self.get_stats()
        print(f"\n================ [Inference Profiling (Step {self.count})] ================")
        print(f"Latency   : {latency:.4f} s | Min: {stats['latency'][0]:.4f} | Max: {stats['latency'][1]:.4f} | Avg: {stats['latency'][2]:.4f} s")
        print(f"Peak VRAM : {peak_vram:.2f} MB | Min: {stats['peak_vram'][0]:.2f} | Max: {stats['peak_vram'][1]:.2f} | Avg: {stats['peak_vram'][2]:.2f} MB")
        print(f"Curr VRAM : {current_vram:.2f} MB | Min: {stats['curr_vram'][0]:.2f} | Max: {stats['curr_vram'][1]:.2f} | Avg: {stats['curr_vram'][2]:.2f} MB")
        print(f"Valid Cnt : {stats['count']} (Total requests: {self.count})")
        print(f"==============================================================\n")

    def get_stats(self):
        if not self.latencies:
            return None
        return {
            "latency": (np.min(self.latencies), np.max(self.latencies), np.mean(self.latencies)),
            "peak_vram": (np.min(self.peak_vrams), np.max(self.peak_vrams), np.mean(self.peak_vrams)),
            "curr_vram": (np.min(self.curr_vrams), np.max(self.curr_vrams), np.mean(self.curr_vrams)),
            "count": len(self.latencies)
        }
        
    def summary(self):
        stats = self.get_stats()
        if stats:
            print(f"\n============== [Final Inference Profiling] ==============")
            print(f"Total Requests   : {self.count}")
            print(f"Profiled Counts  : {stats['count']} (Skipped {self.warmup_steps} warmup steps)")
            print(f"Latency          : Min: {stats['latency'][0]:.4f} | Max: {stats['latency'][1]:.4f} | Avg: {stats['latency'][2]:.4f} s")
            print(f"Peak VRAM        : Min: {stats['peak_vram'][0]:.2f} | Max: {stats['peak_vram'][1]:.2f} | Avg: {stats['peak_vram'][2]:.2f} MB")
            print(f"Curr VRAM        : Min: {stats['curr_vram'][0]:.2f} | Max: {stats['curr_vram'][1]:.2f} | Avg: {stats['curr_vram'][2]:.2f} MB")
            print(f"=========================================================\n")


def analyze_model_stats(model, input_kwargs=None):
    """
    Calculates total parameters and optionally tests fvcore MACs count.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n============== [Model Parameters] ==============")
    print(f"Total Parameters     : {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters : {trainable_params / 1e6:.2f} M")
    print(f"================================================\n")
    
    if input_kwargs is not None:
        try:
            from fvcore.nn import FlopCountAnalysis
            
            # fvcore requires positional arguments. PyTorch JIT fails to trace dynamic dict generation.
            class FvcoreWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, images, image_mask, prompt, state_input, action_mask):
                    return self.model.run_inference(
                        images=images,
                        image_mask=image_mask,
                        prompt=prompt,
                        state_input=state_input,
                        action_mask=action_mask
                    )
            
            wrapper = FvcoreWrapper(model)
            values = (
                input_kwargs.get("images"),
                input_kwargs.get("image_mask"),
                input_kwargs.get("prompt"),
                input_kwargs.get("state_input"),
                input_kwargs.get("action_mask")
            )
            
            flops = FlopCountAnalysis(wrapper, values)
            # Suppress excessive fvcore warnings for unsupported ops
            flops.unsupported_ops_warnings(False)
            
            macs = flops.total()
            print(f"\n============== [Hardware Computing] ==============")
            print(f"fvcore MACs          : {macs / 1e12:.4f} TMACs")
            print(f"================================================\n")
        except ImportError:
            print("[Warning] fvcore not found. Run 'pip install fvcore' for accurate Hardware MACs counting.")
        except Exception as e:
            print(f"[Warning] Could not compute fvcore MACs: {e}")
