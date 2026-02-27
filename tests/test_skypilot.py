"""
Unit tests for SkyPilotLauncher YAML generation.
No actual launch — purely tests generate_task_yaml().
"""
import yaml
import pytest
from gpu_lol.config import EnvironmentSpec
from gpu_lol.skypilot import SkyPilotLauncher


def make_spec(**kwargs) -> EnvironmentSpec:
    defaults = dict(
        gpu_type="A40",
        gpu_count=1,
        vram_required_gb=40,
        base_image="runpod/pytorch:2.1.0-py3.10-cuda12.1-devel-ubuntu22.04",
        requirements=["torch>=2.1.0", "transformers"],
        setup_commands=["pip install -r ~/sky_workdir/requirements.txt --quiet"],
        workload_type="training",
        name="test-cluster",
    )
    defaults.update(kwargs)
    return EnvironmentSpec(**defaults)


def test_yaml_is_valid():
    launcher = SkyPilotLauncher()
    spec = make_spec()
    yaml_str = launcher.generate_task_yaml(spec, "/tmp")
    parsed = yaml.safe_load(yaml_str)
    assert parsed is not None
    assert isinstance(parsed, dict)


def test_yaml_has_required_fields():
    launcher = SkyPilotLauncher()
    spec = make_spec()
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert "name" in parsed
    assert "resources" in parsed
    assert "workdir" in parsed
    assert "setup" in parsed
    assert "run" in parsed
    assert "envs" in parsed


def test_yaml_gpu_accelerator():
    launcher = SkyPilotLauncher()
    spec = make_spec(gpu_type="A40", gpu_count=1)
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert parsed["resources"]["accelerators"] == "A40:1"


def test_yaml_multi_gpu():
    launcher = SkyPilotLauncher()
    spec = make_spec(gpu_type="H100-SXM", gpu_count=4)
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert parsed["resources"]["accelerators"] == "H100-SXM:4"


def test_yaml_docker_image():
    launcher = SkyPilotLauncher()
    image = "runpod/pytorch:2.1.0-py3.10-cuda12.1-devel-ubuntu22.04"
    spec = make_spec(base_image=image)
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert parsed["resources"]["image_id"] == f"docker:{image}"


def test_yaml_no_gpu_type():
    """When gpu_type is None, no accelerators key should appear."""
    launcher = SkyPilotLauncher()
    spec = make_spec(gpu_type=None)
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert "accelerators" not in parsed["resources"]


def test_yaml_multi_provider_fallback():
    launcher = SkyPilotLauncher()
    spec = make_spec()
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    providers = [entry["cloud"] for entry in parsed["resources"]["any_of"]]
    assert "runpod" in providers
    assert "vast" in providers
    assert "lambda" in providers


def test_yaml_setup_commands_joined():
    launcher = SkyPilotLauncher()
    cmds = ["pip install --upgrade pip", "pip install torch"]
    spec = make_spec(setup_commands=cmds)
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert "pip install --upgrade pip" in parsed["setup"]
    assert "pip install torch" in parsed["setup"]


def test_yaml_empty_setup_commands():
    launcher = SkyPilotLauncher()
    spec = make_spec(setup_commands=[])
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert "No setup required" in parsed["setup"]


def test_yaml_envs_contain_workload():
    launcher = SkyPilotLauncher()
    spec = make_spec(workload_type="inference")
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert parsed["envs"]["GPU_ROUTER_MANAGED"] == "1"
    assert parsed["envs"]["GPU_ROUTER_WORKLOAD"] == "inference"


def test_yaml_name_from_spec():
    launcher = SkyPilotLauncher()
    spec = make_spec(name="my-project")
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    assert parsed["name"] == "my-project"


def test_yaml_workdir_is_absolute(tmp_path):
    launcher = SkyPilotLauncher()
    spec = make_spec()
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, str(tmp_path)))

    assert parsed["workdir"] == str(tmp_path)
    assert parsed["workdir"].startswith("/")


def test_yaml_run_keeps_cluster_alive():
    launcher = SkyPilotLauncher()
    spec = make_spec()
    parsed = yaml.safe_load(launcher.generate_task_yaml(spec, "/tmp"))

    # The run command must keep the cluster alive for interactive SSH use
    assert "sleep infinity" in parsed["run"]
