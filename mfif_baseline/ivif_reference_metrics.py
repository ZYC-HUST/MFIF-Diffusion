"""Run the five audited IVIF_ZOO metrics with their original preprocessing.

Only the six required function definitions are loaded from Metric_torch.py. This
avoids importing unrelated metrics and their optional sklearn dependency; the
selected function bodies are not rewritten. PI is not supplied by IVIF_ZOO.
"""

import ast
import hashlib
import math
from pathlib import Path
import re
import subprocess


METRIC_NAMES = ("EN", "AG", "SF", "SD", "SCD")
FUNCTION_NAMES = frozenset(
    {"EN_function", "AG_function", "SF_function", "SD_function", "corr2", "SCD_function"}
)


class IVIFReferenceMetrics:
    """Evaluate saved images using PIL luminance and the reference CPU FP32 code.

    Undefined reference results, including constant-input Pearson correlation,
    raise FloatingPointError. They are never silently replaced by zero.
    """

    def __init__(self, reference_dir=None):
        import torch

        if reference_dir is None:
            reference_dir = Path(__file__).resolve().parents[2] / "IVIF_ZOO" / "Metric"
        self.reference_dir = Path(reference_dir).expanduser().resolve()
        self.source_path = self.reference_dir / "Metric_torch.py"
        source_bytes = self.source_path.read_bytes()
        self.source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        tree = ast.parse(source_bytes, filename=str(self.source_path))
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
        ]
        found = [node.name for node in functions]
        if len(found) != len(FUNCTION_NAMES) or set(found) != FUNCTION_NAMES:
            raise ValueError(
                f"Expected exactly one definition of each required reference function "
                f"in {self.source_path}; found {found}"
            )
        selected = ast.Module(body=functions, type_ignores=[])
        namespace = {"torch": torch}
        exec(compile(selected, str(self.source_path), "exec"), namespace)
        self._functions = {name: namespace[name] for name in FUNCTION_NAMES}
        self._torch = torch
        self.git_commit = self._git_commit()

    def _git_commit(self):
        try:
            result = subprocess.run(
                ["git", "-C", str(self.reference_dir), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        commit = result.stdout.strip()
        return commit if re.fullmatch(r"[0-9a-f]{40,64}", commit) else None

    def evaluate(self, path_a, path_b, path_f):
        """Return EN/AG/SF/SD/SCD Python floats for three same-size image files."""
        import numpy as np
        from PIL import Image

        tensors = []
        for path in (path_a, path_b, path_f):
            with Image.open(path) as image:
                # Match eval_torch.py: RGB -> PIL L -> uint8 array -> float32.
                gray = np.array(image.convert("L"))
            if min(gray.shape) < 2:
                raise ValueError(f"Image height and width must both be at least 2: {path}")
            tensors.append(self._torch.tensor(gray, dtype=self._torch.float32, device="cpu"))
        if not (tensors[0].shape == tensors[1].shape == tensors[2].shape):
            raise ValueError(
                "Source A, source B and fused image must have identical dimensions; "
                f"got {[tuple(tensor.shape) for tensor in tensors]}"
            )
        a, b, fused = tensors
        result = {}
        with self._torch.inference_mode():
            for name in METRIC_NAMES:
                function = self._functions[name + "_function"]
                value = float(function(a, b, fused) if name == "SCD" else function(fused))
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"IVIF_ZOO {name} returned {value} for {path_f}. "
                        "Pearson correlation is undefined for a constant input; "
                        "no replacement score was used."
                    )
                result[name] = value
        return result

    def protocol_info(self):
        """Describe this exact reference implementation and image-file protocol."""
        import PIL

        return {
            "backend": "IVIF_ZOO/Metric/Metric_torch.py, original selected function bodies",
            "reference_dir": str(self.reference_dir),
            "source_file": str(self.source_path),
            "source_sha256": self.source_sha256,
            "git_commit": self.git_commit,
            "loaded_functions": sorted(FUNCTION_NAMES),
            "preprocessing_reference": str(self.reference_dir / "eval_torch.py"),
            "input": "saved image files, decoded independently with PIL.Image.open",
            "grayscale": "PIL.Image.convert('L'); uint8 luminance; no RGB channel averaging",
            "dtype": "torch.float32",
            "device": "cpu",
            "range": [0, 255],
            "normalization": "none",
            "crop_border": 0,
            "resize": "none",
            "EN": "torch.histc(bins=256,min=0,max=255); -sum(p*log2(p+1e-7))",
            "EN_epsilon": 1e-7,
            "AG": "torch.gradient; mean(sqrt((gx**2+gy**2)/2)); edge_order=1",
            "SF": "sqrt(mean(vertical adjacent differences**2)+mean(horizontal adjacent differences**2))",
            "SD": "population standard deviation; denominator H*W",
            "SCD": "corr(F-B,A)+corr(F-A,B); no epsilon or constant-input substitution",
            "nonfinite_policy": "raise FloatingPointError",
            "PI": "not implemented by this reference library; requires a separately specified backend",
            "versions": {"torch": str(self._torch.__version__), "Pillow": PIL.__version__},
            "limitation": "IVIF_ZOO is a separate IVIF toolkit, not a verified Mask-DiFuser paper evaluator",
        }
