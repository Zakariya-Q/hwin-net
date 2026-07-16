++ import pandas as pd
++ import numpy as np
++ from pathlib import Path
++ from sklearn.model_selection import GroupKFold
++ import json
++ 
++ CANONICAL_CSV = Path(r"C:/Users/lenovo/scripts/archive/output_real_hwin_bench_rc1/canonical/HWIN-GRQA-V1-4/observations/observations.csv")
++ OUTPUT_DIR = Path(r"C:/Users/lenovo/hwin_net/data")
++ OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
++ 
++ HWIN_VARIABLES = [
++     "HWIN-VAR-001", "HWIN-VAR-002", "HWIN-VAR-003",
++     "HWIN-VAR-008", "HWIN-VAR-009", "HWIN-VAR-010",
++     "HWIN-VAR-014", "HWIN-VAR-015", "HWIN-VAR-028",
++     "HWIN-VAR-031", "HWIN-VAR-033"
++ ]
++ 
++ SOURCES = ["GEMSTAT", "WATERBASE", "GLORICH", "WQP"]
++ SOURCE_TO_PLATFORM = {src: i for i, src in enumerate(SOURCES)}
++ NUM_VARIABLES = 100
++ 
++ VAR_TO_IDX = {var: i for i, var in enumerate(HWIN_VARIABLES)}
++ 
++ N_FOLDS = 5
++ SEEDS = [42, 123, 456, 789, 999]
++ 
++ print("=" * 60)
++ print("HWIN-Bench Data Preprocessing for HWIN-Net")
++ print("=" * 60)
++ 
++ print("\n[Pass 1] Collecting unique station IDs...")
++ stations = set()
++ chunk_size = 1000000
++ 
++ for i, chunk in enumerate(pd.read_csv(CANONICAL_CSV, usecols=["station_id"], chunksize=chunk_size)):
++     stations.update(chunk["station_id"].unique())
++     if i % 10 == 0:
++         print("  Processed " + str(i * chunk_size) + " rows, " + str(len(stations)) + " unique stations")
++ 
++ stations = sorted(stations)
++ print("Total unique stations: " + str(len(stations)))
++ 
++ with open(OUTPUT_DIR / "stations.json", "w") as f:
++     json.dump({s: i for i, s in enumerate(stations)}, f)
++ 
++ station_folds = {}
++ station_list = np.array(stations)
++ 
++ for seed in SEEDS:
++     gkf = GroupKFold(n_splits=N_FOLDS)
++     folds = np.zeros(len(station_list), dtype=int)
++     for fold_idx, (_, test_idx) in enumerate(gkf.split(station_list, groups=station_list)):
++         folds[test_idx] = fold_idx
++     station_folds[seed] = dict(zip(station_list, folds.tolist()))
++ 
++ for seed in SEEDS:
++     with open(OUTPUT_DIR / ("station_folds_seed_" + str(seed) + ".json"), "w") as f:
++         json.dump(station_folds[seed], f)
++ 
++ print("Folds assigned for " + str(len(SEEDS)) + " seeds")
++ 
++ print("\n[Pass 2] Processing observations and writing parquet files...")
++ 
++ SEED = 42
++ folds = station_folds[SEED]
++ 
++ train_rows = []
++ val_rows = []
++ test_rows = []
++ 
++ def process_chunk(chunk, folds, var_to_idx, source_to_platform, num_variables, train_rows, val_rows, test_rows):
++     grouped = chunk.groupby(["station_id", "timestamp", "source"])
++     
++     for (station_id, timestamp, source), group in grouped:
++         fold = folds.get(station_id, 0)
++         
++         x = np.zeros(num_variables, dtype=np.float32)
++         M_O = np.zeros(num_variables, dtype=np.float32)
++         
++         for _, row in group.iterrows():
++             var = row["variable"]
++             val = row["value"]
++             if var in var_to_idx:
++                 idx = var_to_idx[var]
++                 x[idx] = val
++                 M_O[idx] = 1.0
++         
++         a_idx = source_to_platform.get(source, 0)
++         
++         y = None
++         target_var = "HWIN-VAR-001"
++         if target_var in var_to_idx:
++             tidx = var_to_idx[target_var]
++             if M_O[tidx] == 1:
++                 y = float(x[tidx])
++         
++         row_data = {}
++         for i in range(num_variables):
++             row_data["x_" + str(i)] = x[i]
++         for i in range(num_variables):
++             row_data["M_" + str(i)] = M_O[i]
++         row_data["a_idx"] = a_idx
++         row_data["y"] = y if y is not None else np.nan
++         row_data["station_id"] = station_id
++         row_data["timestamp"] = timestamp
++         row_data["source"] = source
++         
++         if fold == 0:
++             val_rows.append(row_data)
++         elif fold == 1:
++             test_rows.append(row_data)
++         else:
++             train_rows.append(row_data)
++ 
++ total_rows = 0
++ for i, chunk in enumerate(pd.read_csv(CANONICAL_CSV,
++                                        usecols=["station_id", "timestamp", "source", "variable", "value"],
++                                        chunksize=chunk_size)):
++     process_chunk(chunk, folds, VAR_TO_IDX, SOURCE_TO_PLATFORM, NUM_VARIABLES,
++                   train_rows, val_rows, test_rows)
++     total_rows += len(chunk)
++     if i % 50 == 0:
++         print("  Processed " + str(total_rows) + " rows... Train: " + str(len(train_rows)) + ", Val: " + str(len(val_rows)) + ", Test: " + str(len(test_rows)))
++ 
++ print("\nTotal observations processed: " + str(total_rows))
++ print("Train samples: " + str(len(train_rows)))
++ print("Val samples: " + str(len(val_rows)))
++ print("Test samples: " + str(len(test_rows)))
++ 
++ print("\nSaving parquet files...")
++ for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
++     if rows:
++         df_out = pd.DataFrame(rows)
++         df_out.to_parquet(OUTPUT_DIR / (name + ".parquet"), index=False)
++         print("  " + name + ".parquet: " + str(len(df_out)) + " rows, " + str(len(df_out.columns)) + " columns")
++     else:
++         print("  " + name + ".parquet: EMPTY!")
++ 
++ with open(OUTPUT_DIR / "var_to_idx.json", "w") as f:
++     json.dump(VAR_TO_IDX, f, indent=2)
++ with open(OUTPUT_DIR / "source_to_platform.json", "w") as f:
++     json.dump(SOURCE_TO_PLATFORM, f, indent=2)
++ 
++ print("\nDone!")
