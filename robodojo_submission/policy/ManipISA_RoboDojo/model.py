"""Official XPolicyLab wrapper bound to the independent submission checkout."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_pointer = Path(__file__).with_name("submission-root.json")
if _pointer.is_file():
    _submission = Path(json.loads(_pointer.read_text(encoding="utf-8"))["submission_root"]).resolve()
else:
    # Allows importing the source package before installation into XPolicyLab.
    _submission = Path(__file__).resolve().parents[2]
_core = _submission.parent
if not (_submission / "manipisa_robodojo/model.py").is_file() or not (_core / "manipisa/__init__.py").is_file():
    raise RuntimeError("Run robodojo_submission/register_policy.py from the independent checkout first")
for _path in reversed((_submission, _core)):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

import manipisa
import manipisa_robodojo
if Path(manipisa.__file__).resolve().parent != _core / "manipisa":
    raise RuntimeError("A different ManipISA core is already loaded; start a fresh policy process")
if Path(manipisa_robodojo.__file__).resolve().parent != _submission / "manipisa_robodojo":
    raise RuntimeError("A different RoboDojo submission is already loaded; start a fresh policy process")

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from manipisa_robodojo.model import Policy
from manipisa_robodojo.runtime import result_path


class Model(Policy, ModelTemplate):
    def __init__(self, model_cfg):
        model_cfg = dict(model_cfg)
        model_cfg["artifact_dir"] = str(result_path(model_cfg.get("artifact_dir", "results/evaluation")))
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if self.robot_action_dim_info != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
            raise ValueError("The existing ManipISA RoboDojo adapter requires the original dual X5 robot")
        super().__init__(model_cfg)
