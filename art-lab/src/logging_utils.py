import csv, time, subprocess, pathlib

def git_commit_sha():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        return "dirty"

def append_ledger(row:dict, path="logs/run_ledger.csv"):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","seed","cfg","steps","sampler","lora_scales",
            "commit","config_path","notes","grid_path"
        ])
        if write_header: w.writeheader()
        row["timestamp"] = row.get("timestamp") or int(time.time())
        w.writerow(row)