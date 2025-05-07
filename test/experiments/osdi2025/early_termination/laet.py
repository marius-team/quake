# File: laet.py
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import time
import csv
import sys
import math
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Embedded I/O Utilities (Simplified) ---
def _mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r'); d = x[0]; return x.view('float32').reshape(-1, d + 1)[:, 1:]
def _mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r'); d = x[:4].view('int32')[0]; return x.reshape(-1, d + 4)[:, 4:]
def _read_gt_file(fname, format_type, k_load=1):
    gt_list = []
    if format_type == "ivecs":
        if not os.path.exists(fname) or os.path.getsize(fname) == 0:
            return []
        a = np.fromfile(fname, dtype='int32')
        if a.size == 0: return []
        d_gt = a[0]
        if a.size % (d_gt + 1) != 0:
            return []
        data = a.reshape(-1, d_gt + 1)
        for i in range(data.shape[0]):
            count = data[i,0]; num_to_take = min(count, k_load if k_load is not None else count)
            gt_list.append(list(data[i, 1:1 + num_to_take]))
    elif format_type == "tsv":
        if not os.path.exists(fname):
            return []
        with open(fname, 'r', newline='') as f:
            for row in csv.reader(f, delimiter='\t'):
                try:
                    ids = [int(x) for x in row if x.strip()]
                    gt_list.append(ids[:k_load] if k_load is not None else ids)
                except ValueError:
                    pass
    return gt_list
def _write_tsv(data, filename):
    if os.path.exists(filename): os.remove(filename)
    with open(filename, 'w', newline='') as f: csv.writer(f, delimiter='\t').writerows(data)
# --- End I/O ---

class LAETPipeline:
    def __init__(self, vector_dim, laet_artifacts_config):
        self.vector_dim = vector_dim
        self.config = laet_artifacts_config
        self.gbdt_model = None
        self.log_target_prediction = False

        base_dir = Path(self.config["base_dir"])
        self.training_data_output_dir = base_dir / self.config.get("training_data_subdir", "laet_training_data")
        self.model_output_dir = base_dir / self.config.get("model_subdir", "laet_models")

        self.training_data_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name_prefix = f"{self.config['dataset_name_for_files']}_{self.config['index_key_for_files'].replace(',', '_')}"
        _log_suffix_for_naming = "_Log" if self.config.get("model_was_trained_on_log_target", False) else ""
        self.gbdt_model_filename = f"{self.model_name_prefix}_gbdt_model{_log_suffix_for_naming}.txt"
        self.full_gbdt_model_path = self.model_output_dir / self.gbdt_model_filename

    def _find_min_param_faiss(self, faiss_idx, q_vec, gt_set, k_chk, is_hnsw, max_p, init_p=1, step=1):
        if not gt_set: return max_p
        curr_p = init_p
        while curr_p <= max_p:
            if is_hnsw: faiss_idx.hnsw.efSearch = curr_p
            else: faiss_idx.nprobe = curr_p
            try:
                _, I = faiss_idx.search(q_vec, k_chk)
                if I.size > 0 and any(idx in gt_set for idx in I[0] if idx >= 0): return curr_p
            except Exception: pass
            curr_p += step
            if curr_p > max_p and curr_p - step <= max_p :
                if curr_p - step == max_p: continue
                break
        return max_p

    def _generate_training_data_tsv_internal(self, tsv_path, data_paths_cfg, gen_params_cfg, faiss_params_cfg):
        print(f"LAET: Generating training data to {tsv_path}...")
        xb = _mmap_fvecs(data_paths_cfg["base_vectors"]) if data_paths_cfg["base_vectors"].endswith('.fvecs') else _mmap_bvecs(data_paths_cfg["base_vectors"])
        xq_all = _mmap_fvecs(data_paths_cfg["train_query_vectors"]) if data_paths_cfg["train_query_vectors"].endswith('.fvecs') else _mmap_bvecs(data_paths_cfg["train_query_vectors"])

        num_q = gen_params_cfg.get("num_train_queries_to_process", -1)
        xq = xq_all[:num_q] if num_q > 0 else xq_all
        xq_f32 = xq.astype(np.float32)

        gt_all = _read_gt_file(data_paths_cfg["gt_train"], data_paths_cfg["gt_format"], gen_params_cfg.get("k_gt_load_for_target", 1))
        gt = gt_all[:num_q] if num_q > 0 else gt_all
        if len(xq) != len(gt): raise ValueError(f"Query ({len(xq)})/GT ({len(gt)}) length mismatch in training data gen.")

        faiss_idx_key = faiss_params_cfg["faiss_index_key_for_gen"]
        is_hnsw = "hnsw" in faiss_idx_key.lower()
        faiss_idx = faiss.index_factory(self.vector_dim, faiss_idx_key)
        if not faiss_idx.is_trained:
            xt_idx_path = data_paths_cfg.get("training_vectors_for_faiss_index")
            if not xt_idx_path: raise ValueError("FAISS index needs training, but no training vectors path provided.")
            xt_idx = _mmap_fvecs(xt_idx_path) if xt_idx_path.endswith('.fvecs') else _mmap_bvecs(xt_idx_path)
            faiss_idx.train(xt_idx.astype(np.float32))
        faiss_idx.add(xb.astype(np.float32))

        max_p = gen_params_cfg.get("max_efsearch_scan", 512) if is_hnsw else gen_params_cfg.get("max_nprobe_scan", 256)
        if not is_hnsw and hasattr(faiss_idx, 'nlist'): max_p = min(max_p, faiss_idx.nlist)

        k_search_target = gen_params_cfg.get("k_search_for_target", 10)
        param_scan_step = gen_params_cfg.get("param_scan_step", 1)
        out_data = []
        start_gen_time = time.time()
        print(f"LAET Training Data Gen: Max param to scan for optimal_p: {max_p}")
        for i in range(len(xq)):
            opt_p = self._find_min_param_faiss(faiss_idx, xq_f32[i:i+1], set(gt[i]), k_search_target, is_hnsw, max_p, step=param_scan_step)
            out_data.append([opt_p, i] + xq_f32[i].tolist())
            if (i + 1) % 500 == 0 or (i+1) == len(xq):
                elapsed_gen = time.time() - start_gen_time
        _write_tsv(out_data, str(tsv_path))
        # Log distribution of optimal_p found
        if out_data:
            optimal_ps_found = np.array([row[0] for row in out_data])
            print(f"LAET Training Data Gen: Distribution of optimal_p found (target for GBDT): "
                  f"Min={np.min(optimal_ps_found)}, Mean={np.mean(optimal_ps_found):.2f}, Median={np.median(optimal_ps_found)}, Max={np.max(optimal_ps_found)}, "
                  f"StdDev={np.std(optimal_ps_found):.2f}")

        print(f"LAET: Training data saved to {tsv_path}")


    def _train_gbdt_model_internal(self, tsv_path, model_save_path, train_params_cfg):
        print(f"LAET: Training GBDT model from {tsv_path}...")
        data_df = pd.read_csv(tsv_path, sep='\t', header=None)
        target = data_df.iloc[:, 0].values.astype(np.float32)
        features = data_df.iloc[:, 2:].values.astype(np.float32)

        log_target_training = train_params_cfg.get("log_target_training", False)
        if log_target_training: target = np.log2(np.maximum(1, target))
        self.log_target_prediction = log_target_training

        X_tr, X_te, y_tr, y_te = train_test_split(features, target, test_size=train_params_cfg.get("test_size", 0.1), random_state=42)

        lgb_tr_data = lgb.Dataset(X_tr, y_tr)
        lgb_te_data = lgb.Dataset(X_te, y_te, reference=lgb_tr_data)

        lgbm_params = {
            'objective': 'regression_l1', 'metric': 'mae',
            'num_leaves': train_params_cfg.get("num_leaves", 31),
            'learning_rate': train_params_cfg.get("learning_rate", 0.05),
            'feature_fraction': train_params_cfg.get("feature_fraction", 0.9),
            'bagging_fraction': train_params_cfg.get("bagging_fraction", 0.8),
            'bagging_freq': train_params_cfg.get("bagging_freq", 5),
            'verbose': -1, 'n_jobs': -1, 'seed': 42,
            'n_estimators': train_params_cfg.get("n_estimators", 100)
        }
        num_boost_round_val = lgbm_params['n_estimators']

        gbm = lgb.train(lgbm_params, lgb_tr_data, num_boost_round=num_boost_round_val,
                        valid_sets=[lgb_tr_data, lgb_te_data],
                        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(period=0)])

        self.gbdt_model = gbm
        current_log_suffix = "_Log" if self.log_target_prediction else ""
        effective_model_filename = f"{self.model_name_prefix}_gbdt_model{current_log_suffix}.txt"
        effective_model_save_path = self.model_output_dir / effective_model_filename

        gbm.save_model(str(effective_model_save_path))
        self.full_gbdt_model_path = effective_model_save_path
        print(f"LAET: GBDT model saved to {self.full_gbdt_model_path}")

    def get_or_train_model(self, overwrite_model=False,
                           data_paths_config=None,
                           generation_params_config=None,
                           faiss_params_config=None,
                           training_params_config=None
                           ):
        potential_log_target = training_params_config.get("log_target_training", False) if training_params_config else self.config.get("model_was_trained_on_log_target", False)
        _log_suffix_check = "_Log" if potential_log_target else ""
        expected_model_filename_if_trained_now = f"{self.model_name_prefix}_gbdt_model{_log_suffix_check}.txt"
        path_to_check_or_create = self.model_output_dir / expected_model_filename_if_trained_now

        if not overwrite_model and path_to_check_or_create.exists():
            print(f"LAET: Loading existing GBDT model: {path_to_check_or_create}")
            self.gbdt_model = lgb.Booster(model_file=str(path_to_check_or_create))
            self.log_target_prediction = "_Log" in path_to_check_or_create.name
            self.full_gbdt_model_path = path_to_check_or_create
            print(f"LAET: Model loaded. Actual log_target_prediction: {self.log_target_prediction}")
        else:
            print(f"LAET: Training GBDT model (overwrite={overwrite_model}, exists={path_to_check_or_create.exists()})...")
            if not all([data_paths_config, generation_params_config, faiss_params_config, training_params_config]):
                raise ValueError("Missing configurations for LAET model training.")

            training_tsv_filename = f"{self.model_name_prefix}_gbdt_train.tsv"
            full_training_tsv_path = self.training_data_output_dir / training_tsv_filename

            self._generate_training_data_tsv_internal(
                full_training_tsv_path,
                data_paths_config,
                generation_params_config,
                faiss_params_config
            )
            self._train_gbdt_model_internal(
                full_training_tsv_path,
                path_to_check_or_create,
                training_params_config
            )
        if self.vector_dim is None and self.gbdt_model and hasattr(self.gbdt_model, 'num_feature'):
            self.vector_dim = self.gbdt_model.num_feature()

    def _predict_base_search_param(self, query_vector_np):
        if not self.gbdt_model: raise RuntimeError("LAET GBDT model not loaded.")
        if self.vector_dim is None: self.vector_dim = query_vector_np.shape[0]
        if query_vector_np.shape[0] != self.vector_dim:
            raise ValueError(f"Query dim {query_vector_np.shape[0]} != model's {self.vector_dim}")

        features = query_vector_np.astype(np.float32).flatten()
        pred_raw = self.gbdt_model.predict(features.reshape(1, -1))[0]
        pred_val = np.power(2, pred_raw) if self.log_target_prediction else pred_raw
        return max(1, int(round(pred_val)))

    def _compute_recall_single_q(self, res_ids_tensor, gt_set_top_k, k_val_for_recall):
        if not gt_set_top_k or res_ids_tensor is None or res_ids_tensor.numel() == 0 or k_val_for_recall == 0:
            return 0.0
        results_to_check = res_ids_tensor[:min(res_ids_tensor.numel(), k_val_for_recall)].tolist()
        hits = 0
        for res_idx in results_to_check:
            if res_idx in gt_set_top_k and res_idx >= 0:
                hits += 1
        return float(hits) / k_val_for_recall


    def run_inference_for_quake_experiment(self, quake_idx, queries_torch, gt_torch,
                                           target_recall, k_search, base_quake_sp,
                                           num_bs_steps=20, multiplier_range=(0.1, 5.0)):
        if not self.gbdt_model:
            print("LAET Inference Error: GBDT model not loaded.")
            return np.nan, np.nan, np.nan

        nq = queries_torch.shape[0]
        if self.vector_dim is None: self.vector_dim = queries_torch.shape[1]

        gt_sets = [set(gt_torch[i, :min(gt_torch.shape[1], k_search)].tolist()) for i in range(nq)]
        queries_np = queries_torch.numpy()

        multi_low, multi_high = multiplier_range
        best_mult = multi_high
        quake_nlist_cap = quake_idx.nlist() if callable(getattr(quake_idx, 'nlist', None)) else getattr(quake_idx, 'nlist', 2048)

        # --- Log GBDT base_p predictions sample ---
        sample_size_for_log = min(nq, 100)
        sample_base_p_preds = [self._predict_base_search_param(queries_np[i]) for i in range(sample_size_for_log)]
        if sample_base_p_preds:
            print(f"LAET INFERENCE (TargetRecall={target_recall:.2f}): Sample GBDT base_p predictions (first {sample_size_for_log}): "
                  f"Min={np.min(sample_base_p_preds)}, Mean={np.mean(sample_base_p_preds):.2f}, Max={np.max(sample_base_p_preds)}")
        # --- End Log ---

        for bs_run_idx in range(num_bs_steps):
            curr_mult = (multi_low + multi_high) / 2.0
            if abs(multi_low - multi_high) < 1e-4: break
            recall_sum = 0.0

            tune_nq = int(nq * 0.01)
            for i in range(tune_nq):
                base_p = self._predict_base_search_param(queries_np[i])
                final_p = min(max(1, int(round(base_p * curr_mult))), quake_nlist_cap)

                sp = None
                if hasattr(base_quake_sp, 'clone') and callable(base_quake_sp.clone): sp = base_quake_sp.clone()
                else: sp = type(base_quake_sp)()

                sp.nprobe, sp.k = final_p, k_search
                res = quake_idx.search(queries_torch[i:i+1], sp)
                ids = res.ids[0] if res.ids is not None and len(res.ids)>0 and res.ids[0] is not None else torch.empty(0, dtype=torch.long)
                recall_sum += self._compute_recall_single_q(ids, gt_sets[i], k_search)
            avg_rec = recall_sum / tune_nq if tune_nq > 0 else 0.0
            if avg_rec >= target_recall: best_mult, multi_high = curr_mult, curr_mult
            else: multi_low = curr_mult

        print(f"LAET INFERENCE (TargetRecall={target_recall:.2f}): Determined best_mult = {best_mult:.3f}")

        final_recall_sum, total_time_ns, sum_nprobes = 0.0, 0, 0
        final_p_values_sample = []

        for i in range(nq):
            base_p = self._predict_base_search_param(queries_np[i])
            final_p = min(max(1, int(round(base_p * best_mult))), quake_nlist_cap)
            sum_nprobes += final_p
            if i < sample_size_for_log: # Log sample of final_p
                final_p_values_sample.append(final_p)

            sp_final = None
            if hasattr(base_quake_sp, 'clone') and callable(base_quake_sp.clone): sp_final = base_quake_sp.clone()
            else: sp_final = type(base_quake_sp)()

            sp_final.nprobe, sp_final.k = final_p, k_search

            t_s = time.perf_counter_ns()
            res = quake_idx.search(queries_torch[i:i+1], sp_final)
            total_time_ns += (time.perf_counter_ns() - t_s)
            ids = res.ids[0] if res.ids is not None and len(res.ids)>0 and res.ids[0] is not None else torch.empty(0, dtype=torch.long)
            final_recall_sum += self._compute_recall_single_q(ids, gt_sets[i], k_search)

        if final_p_values_sample:
            print(f"LAET INFERENCE (TargetRecall={target_recall:.2f}): Sample final_p values used (first {len(final_p_values_sample)}): "
                  f"Min={np.min(final_p_values_sample)}, Mean={np.mean(final_p_values_sample):.2f}, Max={np.max(final_p_values_sample)}")


        avg_rec_f = final_recall_sum / nq if nq > 0 else 0.0
        avg_time_ms_f = (total_time_ns / nq) / 1e6 if nq > 0 else 0.0
        avg_nprobe_f = sum_nprobes / nq if nq > 0 else 0.0


        return avg_nprobe_f, avg_rec_f, avg_time_ms_f
