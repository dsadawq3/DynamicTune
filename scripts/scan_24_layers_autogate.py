import sys, os
sys.path.insert(0, r".")
import pickle, json, time, torch, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from faytuna_flow.knots import compute_subspace_entropy
from scripts.run_qwen35_transfer import TRAIN_PROMPTS, collect_hidden_states, build_multi_chart_atlas

with open('runs/qwen35_instruct_anchor_head/teacher_traces_cache.pkl', 'rb') as f:
    t_traces = pickle.load(f)

tok = AutoTokenizer.from_pretrained('C:/models/Qwen3.5-0.8B')
s_model = AutoModelForCausalLM.from_pretrained('C:/models/Qwen3.5-0.8B', dtype=torch.float32, low_cpu_mem_usage=True)
s_traces = collect_hidden_states(s_model, tok, TRAIN_PROMPTS, device='cpu', batch_size=4)

n_s = 24
n_t = 32
print('Layer | Teacher | Multi-Chart Atlas Entropy | Auto-Gate Verdict')
print('-' * 65)
candidates = []
for s_idx in range(n_s):
    t_idx = int(round(s_idx * (n_t - 1) / (n_s - 1)))
    s_in = np.concatenate([tr[s_idx] for tr in s_traces], axis=0)
    s_out = np.concatenate([tr[s_idx + 1] for tr in s_traces], axis=0)
    t_in = np.concatenate([tr[t_idx] for tr in t_traces], axis=0)
    t_out = np.concatenate([tr[t_idx + 1] for tr in t_traces], axis=0)
    
    atlas_in = build_multi_chart_atlas(s_in, t_in, n_charts=4, random_state=42 + s_idx)
    atlas_out = build_multi_chart_atlas(s_out, t_out, n_charts=4, random_state=100 + s_idx)
    
    t_in_proj = atlas_in.project_teacher_to_student(t_in, s_in)
    t_out_proj = atlas_out.project_teacher_to_student(t_out, s_out)
    
    flow_res = (t_out_proj - t_in_proj) - (s_out - s_in)
    ent = compute_subspace_entropy(flow_res)
    is_cand = ent < 0.88
    if is_cand:
        candidates.append((s_idx, t_idx, ent))
    verdict = "*** TRANSFERABLE CANDIDATE ***" if is_cand else "Knot / Polysemantic (Skip)"
    print(f"{s_idx:5d} | {t_idx:7d} | {ent:25.4f} | {verdict}")

print(f"\nTotal Auto-Gate Transferable Candidates: {len(candidates)}")
for c in candidates:
    print(f"  Layer {c[0]:02d} -> Teacher {c[1]:02d} (Spectral Entropy: {c[2]:.4f})")
