import json
import shutil
from pathlib import Path

import numpy as np
from vaml.buffer import UpdateBatch


def create_dummy_data(
    data_dir: Path, num_files: int, episodes_per_file: int, seq_length: int
):
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_files):
        batch = UpdateBatch(
            context_length=np.zeros((episodes_per_file,), dtype=np.int32),
            context=np.zeros((episodes_per_file, seq_length), dtype=np.int32),
            log_probs=np.zeros((episodes_per_file, seq_length - 1), dtype=np.float32),
            rewards=np.zeros((episodes_per_file, seq_length), dtype=np.float32),
            policy_mask=np.zeros((episodes_per_file, seq_length), dtype=np.bool_),
            turn_counts=np.zeros((episodes_per_file,), dtype=np.int32),
            turn_start_positions=np.zeros((episodes_per_file, 1), dtype=np.int32),
        )
        batch.save_npz(data_dir / f"episodes_{i}.npz")


def test_loading():
    data_dir = Path("./test_offline_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    num_files = 3
    episodes_per_file = 10
    seq_length = 32
    create_dummy_data(data_dir, num_files, episodes_per_file, seq_length)

    config_dict = {
        "base_model": "gpt2",  # Use a small model if possible, but load_base_model might be complex
        "lora": {"mlp": True, "attn": True, "rank": 8},
        "logger": {"project_name": "test_rl"},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "loss": {
            "gae_lambda": 0.95,
            "gae_discount": 0.99,
            "vf_coef": 0.5,
            "pg_clip_high": 1.2,
            "pg_clip_low": 0.8,
        },
        "env": {"name": "arithmetic", "max_x": 10, "max_y": 10},
        "eval_envs": 1,
        "update_envs": 5,  # batch size for buffer take_batch
        "max_seq_length": seq_length,
        "total_update_episodes": 100,
        "checkpoint_every": 10,
        "offline_data_url": str(data_dir.absolute()),
    }

    config_path = Path("./test_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    # We need to mock load_base_model and update_step or use real ones if they are fast
    # For now, let's just see if it runs up to the point of failure or success
    # I'll use a try-except to catch errors if dependencies are missing in this env

    print("Running train_value_cli with dummy data...")
    try:
        # We might need to mock things if load_base_model fails
        # But let's try to run it first.
        # experiment = Experiment.from_config_file(str(config_path))
        # ...
        pass
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # shutil.rmtree(data_dir)
        # config_path.unlink()
        pass


if __name__ == "__main__":
    test_loading()
