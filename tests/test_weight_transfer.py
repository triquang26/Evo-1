import torch
import argparse

def verify_weight_transfer(teacher_ckpt, student_ckpt, action_head_map="0:0,2:1,4:2,6:3", vlm_map="0:0,2:1,4:2,6:3,8:4,11:5,13:6"):
    print(f"Loading Teacher: {teacher_ckpt}")
    teacher_state = torch.load(teacher_ckpt, map_location='cpu')
    if "module" in teacher_state: teacher_state = teacher_state["module"]
        
    print(f"Loading Student: {student_ckpt}")
    student_state = torch.load(student_ckpt, map_location='cpu')
    if "module" in student_state: student_state = student_state["module"]
        
    action_head_mapping = {int(k): int(v) for k,v in (pair.split(":") for pair in action_head_map.split(","))}
    vlm_mapping = {int(k): int(v) for k,v in (pair.split(":") for pair in vlm_map.split(","))}

    failed = False
    
    # Check Action Head
    print(f"\n--- Checking Action Head Weights ---")
    for t_idx, s_idx in action_head_mapping.items():
        matched = 0
        for k, v in teacher_state.items():
            if f"action_head.transformer_blocks.{t_idx}." in k:
                s_key = k.replace(f"action_head.transformer_blocks.{t_idx}.", f"action_head.transformer_blocks.{s_idx}.")
                if s_key not in student_state:
                    print(f"❌ Missing key in student: {s_key}")
                    failed = True
                    continue
                
                # Compare tensors
                if not torch.equal(v, student_state[s_key]):
                    print(f"❌ Value mismatch for layer {s_key}")
                    failed = True
                else:
                    matched += 1
        print(f"✅ Teacher Layer {t_idx} -> Student Layer {s_idx}: {matched} parameters perfectly matched.")

    # Check VLM
    print(f"\n--- Checking VLM Weights ---")
    for t_idx, s_idx in vlm_mapping.items():
        matched = 0
        for k, v in teacher_state.items():
            s_key = None
            if f"embedder.model.language_model.model.layers.{t_idx}." in k:
                s_key = k.replace(f"embedder.model.language_model.model.layers.{t_idx}.", f"embedder.model.language_model.model.layers.{s_idx}.")
            if f"embedder.model.language_model.layers.{t_idx}." in k:
                s_key = k.replace(f"embedder.model.language_model.layers.{t_idx}.", f"embedder.model.language_model.layers.{s_idx}.")
            
            if s_key is not None:
                if s_key not in student_state:
                    print(f"❌ Missing key in student: {s_key}")
                    failed = True
                    continue
                if not torch.equal(v, student_state[s_key]):
                    print(f"❌ Value mismatch for VLM layer {s_key}")
                    failed = True
                else:
                    matched += 1
        print(f"✅ Teacher VLM Layer {t_idx} -> Student VLM Layer {s_idx}: {matched} parameters perfectly matched.")

    if not failed:
        print("\n🎉 SUCCESS: All mapped parameters are purely identical to the Teacher!")
        return True
    else:
        print("\n⛔ FAILURE: Weights do not perfectly match.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True)
    args = parser.parse_args()
    verify_weight_transfer(args.teacher_ckpt, args.student_ckpt)
