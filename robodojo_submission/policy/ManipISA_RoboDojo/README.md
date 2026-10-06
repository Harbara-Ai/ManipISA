# ManipISA_RoboDojo

Evaluation-only XPolicyLab adapter for dual ARX X5 (`arx_x5`, `ee`, one environment). It uses the isolated `robodojo_submission` package and the unchanged pinned ManipISA core from the same checkout. The policy is fixed to native WSL Codex `gpt-6-astra` / `high` with the existing login.

Use `robodojo_submission/register_policy.py --xpolicylab /path/to/XPolicyLab` to add this policy. Registration writes a machine-local `submission-root.json`; do not publish that file or copy it between machines. Existing conflicting files are never overwritten. `model.py` checks imported module locations before constructing the policy. See `robodojo_submission/README.md` in the submission repository for setup, feedback semantics, tests and Windows/WSL instructions.

The official evaluator owns physics, IK, episode termination and scoring. The policy accepts only the declared three-camera RGB and robot-state profile. Gripper values are control targets; MOVE completion does not establish task success. Contact/wrench/object truth and evaluator labels are unavailable to the agent.

The shell entrypoints follow XPolicyLab's standard arguments. The environment client may run separately from the policy server. No policy-side simulator or training/checkpoint conversion is required. This closed-model route still needs committee confirmation for official Verified publication requirements; successful local transport tests are not benchmark scores.
