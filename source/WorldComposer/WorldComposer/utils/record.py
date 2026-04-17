from pathlib import Path
import time
import json
import numpy as np


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_next_experiment_path_with_gap(base_path: Path) -> Path:
    """Return the first available numeric experiment folder, including gaps."""
    base_path.mkdir(parents=True, exist_ok=True)

    indices = set()
    for folder in base_path.iterdir():
        if folder.is_dir():
            try:
                indices.add(int(folder.name))
            except ValueError:
                continue

    folder_index = 1
    while folder_index in indices:
        folder_index += 1

    return base_path / f"{folder_index:03d}"


def _ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    return obj


def append_episode_initial_pose(jsonl_path, episode_idx, object_initial_pose):
    object_initial_pose = _ndarray_to_list(object_initial_pose)
    rec = {"episode_idx": episode_idx, "object_initial_pose": object_initial_pose}
    with open(jsonl_path, "a") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
