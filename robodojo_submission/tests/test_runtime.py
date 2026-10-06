"""Runtime selection must preserve WSL identity and explicit evaluation settings."""
import os
import json
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from manipisa_robodojo.runtime import resolve_codex, result_path, SUBMISSION_ROOT
from manipisa_robodojo.agent import codex_command
from register_policy import register


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.codex = self.root / "codex"
        self.codex.write_text("#!/bin/sh\nexit 0\n")
        self.codex.chmod(0o700)
        self.settings = self.root / "config.toml"
        self.settings.write_text('model="gpt-6-luna"\nmodel_reasoning_effort="xhigh"\n')
        env = patch.dict(os.environ, {"CODEX_HOME": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def resolve(self, **kwargs):
        return resolve_codex({"codex_path": str(self.codex), **kwargs})

    def test_fixed_submission_does_not_inherit_or_rewrite_global_model(self):
        before = self.settings.read_bytes()
        result = self.resolve(reasoning_effort="high")
        self.assertEqual(result["model"], "gpt-6-astra")
        self.assertEqual(result["reasoning_effort"], "high")
        self.assertEqual(result["configured_reasoning_effort"], "xhigh")
        self.assertEqual(self.settings.read_bytes(), before)
        self.assertEqual(self.resolve()["reasoning_effort"], "high")
        for override in ({"model": "gpt-6-luna"}, {"reasoning_effort": "medium"}):
            with self.assertRaisesRegex(ValueError, "fixed"):
                self.resolve(**override)

    def test_windows_codex_cannot_be_selected(self):
        with self.assertRaisesRegex(ValueError, "Windows Codex"):
            resolve_codex({"codex_path": str(self.root / "codex.exe")})
        with patch("manipisa_robodojo.runtime.sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "inside WSL"):
                self.resolve()

    def test_missing_global_model_keeps_submission_identity_and_custom_provider_is_rejected(self):
        self.settings.write_text('model_reasoning_effort="high"\n')
        self.assertEqual(self.resolve()["model"], "gpt-6-astra")
        self.settings.write_text('model="custom-model"\nmodel_provider="custom"\n')
        with self.assertRaisesRegex(ValueError, "Custom Codex provider"):
            self.resolve()

    def test_only_nonsecret_config_is_read(self):
        original = Path.read_text
        def checked(path, *args, **kwargs):
            if path.name in ("auth.json", "credentials.json"):
                raise AssertionError("Credential file must not be read by the resolver")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", checked):
            self.assertEqual(self.resolve()["model"], "gpt-6-astra")

    def test_child_command_uses_four_mcp_tools_and_episode_timeout(self):
        command = codex_command(self.resolve(), self.root / "work", "http://[::1]:9000", "fixture", 1200)
        overrides = {}
        for i, value in enumerate(command[:-1]):
            if value == "-c":
                key, _, raw = command[i+1].partition("=")
                if key != "otel.exporter":
                    overrides[key] = json.loads(raw)
        self.assertEqual(overrides["mcp_servers.experiment.tool_timeout_sec"], 1260)
        arguments = overrides["mcp_servers.experiment.args"]
        self.assertEqual(arguments[-2:], ["--timeout", "1260"])
        self.assertEqual(Path(arguments[0]).resolve().parents[2], SUBMISSION_ROOT.parent)
        self.assertEqual(overrides["mcp_servers.experiment.enabled_tools"],
                         ["start_task", "observe", "read_api", "execute_python"])
        disabled = {command[i+1] for i, value in enumerate(command[:-1]) if value == "--disable"}
        self.assertTrue({"shell_tool", "view_image", "computer_use", "multi_agent", "plugins", "apps"} <= disabled)
        self.assertNotIn("code_mode_host", disabled)  # Required to orchestrate MCP on Codex 0.160.x.
        self.assertEqual(command[command.index("--sandbox")+1], "read-only")
        self.assertEqual(overrides["web_search"], "disabled")
        self.assertEqual(overrides["forced_login_method"], "chatgpt")

    def test_output_must_stay_in_submission_results(self):
        self.assertEqual(result_path("results/example"), SUBMISSION_ROOT / "results/example")
        for path in (self.root, "../original", "results/../../escape"):
            with self.assertRaisesRegex(ValueError, "Outputs must stay"):
                result_path(path)

    def test_registration_refuses_all_writes_if_any_destination_conflicts(self):
        checkout = self.root / "XPolicyLab"
        checkout.mkdir()
        (checkout / "setup_policy_server.py").touch()
        target = register(checkout)
        pointer = json.loads((target / "submission-root.json").read_text())
        self.assertEqual(Path(pointer["submission_root"]), SUBMISSION_ROOT)
        self.assertTrue((target / "setup_eval_policy_server.sh").is_file())
        self.assertTrue((target / "setup_eval_env_client.sh").is_file())
        marker = target / "deploy.yml"
        marker.write_text("user change\n")
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        with self.assertRaises(FileExistsError):
            register(checkout)
        self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, before)

    def test_registered_policy_imports_independent_checkout_without_pythonpath(self):
        checkout = self.root / "deployment" / "XPolicyLab"
        checkout.mkdir(parents=True)
        (checkout / "setup_policy_server.py").touch()
        target = register(checkout)
        # Minimal official interfaces isolate import/bootstrap behavior from
        # heavyweight simulator dependencies. No model request is started.
        code = """
import importlib.util, json, pathlib, sys, types
for name in ('XPolicyLab', 'XPolicyLab.model_template', 'XPolicyLab.utils', 'XPolicyLab.utils.process_data'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['XPolicyLab.model_template'].ModelTemplate = type('ModelTemplate', (), {})
sys.modules['XPolicyLab.utils.process_data'].get_robot_action_dim_info = lambda _: {'arm_dim': [6,6], 'ee_dim': [1,1]}
spec = importlib.util.spec_from_file_location('registered_policy', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.Model({'action_type': 'ee', 'env_cfg_type': 'arx_x5', 'codex_path': sys.argv[2]})
import manipisa, manipisa_robodojo
print(json.dumps({'core': str(pathlib.Path(manipisa.__file__).resolve().parent),
                  'addon': str(pathlib.Path(manipisa_robodojo.__file__).resolve().parent),
                  'model': model.runner.settings['model'], 'effort': model.runner.settings['reasoning_effort']}))
"""
        completed = subprocess.run([sys.executable, "-B", "-c", code, str(target / "model.py"), str(self.codex)],
                                   cwd=self.root, env=dict(os.environ, PYTHONPATH=""), text=True,
                                   capture_output=True, check=True, timeout=30)
        result = json.loads(completed.stdout)
        self.assertEqual(Path(result["core"]), SUBMISSION_ROOT.parent / "manipisa")
        self.assertEqual(Path(result["addon"]), SUBMISSION_ROOT / "manipisa_robodojo")
        self.assertEqual((result["model"], result["effort"]), ("gpt-6-astra", "high"))

    def test_linux_client_timeout_override_preserves_explicit_values(self):
        path = SUBMISSION_ROOT / "policy/ManipISA_RoboDojo/run_eval_client.py"
        spec = importlib.util.spec_from_file_location("submission_client_entry", path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        class Original:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        module = SimpleNamespace(WsModelClient=Original)
        entry.install_transport_timeout(module, 960)
        self.assertEqual(module.WsModelClient(url="fixture").kwargs["request_timeout_s"], 960)
        self.assertEqual(module.WsModelClient(request_timeout_s=2).kwargs["request_timeout_s"], 2)
        with self.assertRaises(ValueError):
            entry.install_transport_timeout(module, float("inf"))

    @unittest.skipUnless(sys.platform == "linux", "Linux shell entrypoint")
    def test_actual_linux_client_shell_passes_official_arguments_through_wrapper(self):
        import shlex
        checkout = self.root / "RoboDojo" / "XPolicyLab"
        checkout.mkdir(parents=True)
        (checkout / "setup_policy_server.py").touch()
        target = register(checkout)
        binary = self.root / "bin"
        binary.mkdir()
        conda_base = self.root / "conda"
        conda_init = conda_base / "etc/profile.d/conda.sh"
        conda_init.parent.mkdir(parents=True)
        conda_init.write_text("conda() { :; }\n")
        conda = binary / "conda"
        conda.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(str(conda_base)) + "\n")
        conda.chmod(0o700)
        output = self.root / "shell-argv.json"
        python = binary / "python"
        python.write_text("#!" + sys.executable + "\nimport json,os,sys\n"
                          "with open(os.environ['TEST_CAPTURE'], 'w') as f:\n"
                          "    json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'eval_num': os.environ.get('EVAL_NUM')}, f)\n")
        python.chmod(0o700)
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                   TEST_CAPTURE=str(output), EVAL_ENV_TYPE="sim", EVAL_NUM="native")
        subprocess.run(["bash", str(target / "setup_eval_env_client.sh"), "RoboDojo", "push_T", "codex",
                        "arx_x5", "ee", "2", "0", "fixture-env", "fixture-label", "19000", "::1"],
                       env=env, capture_output=True, text=True, check=True, timeout=5)
        capture = json.loads(output.read_text())
        args = capture["argv"]
        self.assertEqual(args[:2], ["-u", str(target / "run_eval_client.py")])
        for flag, value in (("--task_name", "push_T"), ("--seed", "2"), ("--num_envs", "1"),
                            ("--policy_server_url", "ws://[::1]:19000")):
            self.assertEqual(args[args.index(flag)+1], value)
        self.assertEqual(Path(capture["cwd"]), checkout.parent)
        self.assertEqual(capture["eval_num"], "native")


if __name__ == "__main__":
    unittest.main()
