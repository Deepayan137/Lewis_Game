import os
import json
from datetime import datetime
import wandb

class PredictionLogger:
    def __init__(self, log_path=None, wandb_enabled=True):
        self.entries = []
        self.wandb_enabled = wandb_enabled and wandb.run is not None
        self.log_path = log_path or os.getenv("LOG_PATH") or "predictions_log.json"

        if self.wandb_enabled:
            self.wandb_table = wandb.Table(columns=["timestamp", "reward", "content", "solution"])
    
    def log(self, reward, content, sol):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "reward": reward,
            "content": content,
            "solution": sol
        }
        self.entries.append(entry)

        if self.wandb_enabled:
            self.wandb_table.add_data(timestamp, reward, content, sol)

    def flush(self):
        # Write to JSON
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)

        # Optionally log to W&B
        if self.wandb_enabled:
            wandb.log({"prediction_table": self.wandb_table})
