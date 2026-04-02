from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import triton_python_backend_utils as pb_utils  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    pb_utils = None


def _add_source_to_path() -> None:
    candidates = []

    env_root = os.environ.get("LOCALIZED_WEATHER_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend([
        Path("/projects3/home/flag0220/LocalizedWeather"),
        Path("/workspace"),
        Path("/workspace/Source"),
        Path(__file__).resolve().parents[3],
        Path(__file__).resolve().parents[3] / "Source",
    ])

    for root in candidates:
        source_dir = root if (root / "Modules").exists() else root / "Source"
        mpnn_file = source_dir / "Modules" / "GNN" / "MPNN.py"
        if mpnn_file.exists():
            source_path = str(source_dir)
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
            return

    raise RuntimeError(
        "Could not locate the LocalizedWeather Source directory. Set LOCALIZED_WEATHER_ROOT "
        "or mount the project so Source/Modules/GNN/MPNN.py is available."
    )


_add_source_to_path()


def _load_mpnn_class():
    candidate_paths = [
        Path(os.environ.get("LOCALIZED_WEATHER_ROOT", "")) / "Source" / "Modules" / "GNN" / "MPNN.py",
        Path("/projects3/home/flag0220/LocalizedWeather/Source/Modules/GNN/MPNN.py"),
        Path("/workspace/Source/Modules/GNN/MPNN.py"),
    ]

    for mpnn_file in candidate_paths:
        if mpnn_file.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("localizedweather_mpnn", mpnn_file)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.MPNN

    raise RuntimeError("Could not import MPNN from the LocalizedWeather source tree.")


MPNN = _load_mpnn_class()


class TritonPythonModel:
    def initialize(self, args):
        repo_root = Path(args["model_repository"])
        model_name = args["model_name"]
        model_version = args["model_version"]
        self.model_dir = None
        for candidate in (
            repo_root / model_name / model_version,
            repo_root / model_version,
            repo_root.parent / model_name / model_version,
            repo_root.parent / model_version,
        ):
            if (candidate / "best_model.pt").exists():
                self.model_dir = candidate
                break

        if self.model_dir is None:
            raise FileNotFoundError(
                f"Could not locate best_model.pt under any expected Triton model path based on {repo_root}"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = self.model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

        hidden_dim = state_dict["embedding_mlp.0.weight"].shape[0]
        n_node_features_m = state_dict["embedding_mlp.0.weight"].shape[1] - 2
        n_node_features_e = state_dict["gnn_ex_1.ex_embed_net_1.0.weight"].shape[1] - 2
        n_out_features = state_dict["output_mlp.2.weight"].shape[0]
        n_passing = 0
        while f"gnn_layers.{n_passing}.message_net_1.0.weight" in state_dict:
            n_passing += 1

        self.model = MPNN(
            n_passing=n_passing,
            lead_hrs=0,
            n_node_features_m=n_node_features_m,
            n_node_features_e=n_node_features_e,
            n_out_features=n_out_features,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def execute(self, requests):
        responses = []
        with torch.no_grad():
            for request in requests:
                madis_x = self._get_tensor(request, "madis_x", dtype=np.float32)
                madis_lon = self._get_tensor(request, "madis_lon", dtype=np.float32)
                madis_lat = self._get_tensor(request, "madis_lat", dtype=np.float32)
                edge_index = self._get_tensor(request, "edge_index", dtype=np.int64)
                ex_lon = self._get_tensor(request, "ex_lon", dtype=np.float32)
                ex_lat = self._get_tensor(request, "ex_lat", dtype=np.float32)
                ex_x = self._get_tensor(request, "ex_x", dtype=np.float32)
                edge_index_e2m = self._get_tensor(request, "edge_index_e2m", dtype=np.int64)

                pred = self.model(
                    madis_x.to(self.device),
                    madis_lon.to(self.device),
                    madis_lat.to(self.device),
                    edge_index.to(self.device),
                    ex_lon.to(self.device),
                    ex_lat.to(self.device),
                    ex_x.to(self.device),
                    edge_index_e2m.to(self.device),
                )

                pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
                responses.append(pb_utils.InferenceResponse([
                    pb_utils.Tensor("pred", pred_np)
                ]))

        return responses

    @staticmethod
    def _get_tensor(request, name: str, dtype):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            raise ValueError(f"Missing required input tensor: {name}")
        array = tensor.as_numpy()
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        return torch.from_numpy(array)

    def finalize(self):
        pass