"""Resolve filesystem paths for the zero-shot pipeline inputs and outputs."""

import os
from pathlib import Path


def _resolve_path(value):
    if not value:
        return None
    return Path(value).expanduser().resolve()


class ZeroShotPaths:
    def __init__(self, data_root, workspace_root):
        self.data_root = Path(data_root)
        self.workspace_root = Path(workspace_root)

    @classmethod
    def from_env(cls):
        data_root = _resolve_path(os.environ.get("TEXT2SHAPE_DATA_ROOT"))
        workspace_root = _resolve_path(os.environ.get("ZERO_SHOT_WORKDIR"))
        repo_root = Path(__file__).resolve().parents[2]

        if not data_root:
            raise ValueError("Set TEXT2SHAPE_DATA_ROOT to the Text2Shape asset directory.")
        if repo_root in data_root.parents or data_root == repo_root:
            raise ValueError("TEXT2SHAPE_DATA_ROOT must live outside the repository tree.")

        if not workspace_root:
            workspace_root = Path.cwd() / "zero_shot_runs"
        workspace_root.mkdir(parents=True, exist_ok=True)

        return cls(data_root, workspace_root)

    @property
    def mesh_dir(self):
        return self.data_root / "text2shape_mesh"

    @property
    def render_dir(self):
        return self.workspace_root / "renders"

    @property
    def captions_jsonl(self):
        return self.workspace_root / "captions_ranked.jsonl"

    @property
    def retrieval_index(self):
        return self.workspace_root / "cap3d_zero_shot_index.csv"
