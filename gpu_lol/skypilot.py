import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import yaml
from .config import EnvironmentSpec


def _sky() -> str:
    """Resolve the sky binary — checks PATH first, then the current venv."""
    if (found := shutil.which("sky")):
        return found
    venv_sky = Path(sys.executable).parent / "sky"
    if venv_sky.exists():
        return str(venv_sky)
    raise FileNotFoundError(
        "sky not found. Install SkyPilot: pip install 'skypilot[runpod,vast,lambda]'"
    )


class SkyPilotLauncher:

    def generate_task_yaml(self, spec: EnvironmentSpec, workdir: str = ".") -> str:
        """
        Generate SkyPilot task YAML from EnvironmentSpec.
        This YAML is what SkyPilot uses to provision the cluster.
        """
        workdir = str(Path(workdir).resolve())

        resources = {
            "image_id": f"docker:{spec.base_image}",
        }

        if spec.gpu_type:
            resources["accelerators"] = f"{spec.gpu_type}:{spec.gpu_count}"

        # Allow SkyPilot to try multiple providers for best price/availability
        resources["any_of"] = [
            {"cloud": "runpod"},
            {"cloud": "vast"},
            {"cloud": "lambda"},
        ]

        # Generate symlink setup commands for large assets (datasets, model weights)
        symlink_cmds = []
        for link in getattr(spec, 'asset_links', []):
            if ":" in link:
                remote, local = link.split(":", 1)
                remote = remote.strip()
                local = local.strip()
                parent = str(Path(local).parent)
                # mkdir parent, remove existing file/dir at local path, create symlink
                symlink_cmds.append(
                    f"mkdir -p {parent} && ([ -L {local} ] || [ ! -e {local} ]) && "
                    f"ln -sf {remote} {local} || echo 'Asset already exists at {local}'"
                )

        task = {
            "name": spec.name,
            "resources": resources,
            "workdir": workdir,  # SkyPilot syncs this to ~/sky_workdir/ on cluster
            "setup": "\n".join(symlink_cmds + (spec.setup_commands or [])) if (symlink_cmds or spec.setup_commands) else "echo 'No setup required'",
            # Keep cluster alive for interactive SSH use
            "run": "echo '✅ gpu-lol: environment ready' && sleep infinity",
            "envs": {
                "GPU_LOL_MANAGED": "1",
                "GPU_LOL_WORKLOAD": spec.workload_type,
            }
        }

        return yaml.dump(task, default_flow_style=False, allow_unicode=True)

    def launch(self, spec: EnvironmentSpec, cluster_name: str, workdir: str = ".", detach: bool = False, autostop_hours: float | None = None) -> str:
        """
        Launch a SkyPilot cluster.
        Returns cluster_name on success.
        If detach=True, returns immediately without waiting for setup to complete.
        If autostop_hours is set, cluster auto-stops after that many idle hours.
        Raises subprocess.CalledProcessError on failure.
        """
        task_yaml = self.generate_task_yaml(spec, workdir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="gpu-lol-") as f:
            f.write(task_yaml)
            task_file = f.name

        cmd = [_sky(), "launch", "-c", cluster_name, "--yes", "--detach-run", task_file]
        if autostop_hours is not None:
            idle_minutes = max(1, int(autostop_hours * 60))
            cmd += ["--idle-minutes-to-autostop", str(idle_minutes)]

        try:
            if detach:
                # Fire and forget — stream output to a log file, return immediately
                log_path = Path(tempfile.gettempdir()) / f"gpu-lol-{cluster_name}.log"
                with open(log_path, "w") as log:
                    subprocess.Popen(cmd, stdout=log, stderr=log)
                print(f"  Logs: {log_path}")
            else:
                # Stream output live but also capture for error detection
                import sys
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                output_lines = []
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    output_lines.append(line)
                process.wait()
                if process.returncode != 0:
                    full_output = "".join(output_lines)
                    # Raise with output embedded so cli.py can parse it
                    err = subprocess.CalledProcessError(process.returncode, cmd)
                    err.output = full_output
                    raise err
        finally:
            # Only unlink immediately for blocking launch; detached process still needs the file
            if not detach:
                Path(task_file).unlink(missing_ok=True)

        return cluster_name

    def exec(self, cluster_name: str, command: str, timeout: int = 60) -> tuple[str, int]:
        """
        Execute a shell command on a running cluster via SSH (inline, not queued).
        Uses SkyPilot's generated SSH config which has the correct port + key.
        Returns (combined_output, exit_code).
        """
        try:
            ssh_config = Path.home() / ".sky" / "generated" / "ssh" / cluster_name
            if ssh_config.exists():
                # Use sky's generated config — has correct port, key, and host
                ssh_cmd = ["ssh", "-F", str(ssh_config), cluster_name, command]
            else:
                # Fallback: get IP and try default port with user's keys
                ip = self._get_ip(cluster_name)
                if not ip:
                    return f"Could not resolve IP for cluster '{cluster_name}'", 1
                ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", f"root@{ip}", command]

            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return output, result.returncode
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s", 1
        except Exception as e:
            return str(e), 1

    def _get_ip(self, cluster_name: str) -> str:
        """Get the public IP of a running cluster."""
        result = subprocess.run(
            [_sky(), "status", "--ip", cluster_name],
            capture_output=True, text=True
        )
        return result.stdout.strip()

    def get_ssh_info(self, cluster_name: str) -> dict:
        """Get SSH connection details for a cluster."""
        ip = self._get_ip(cluster_name)
        ssh_config = Path.home() / ".sky" / "generated" / "ssh" / cluster_name
        # Parse port from sky's generated config
        port = 22
        if ssh_config.exists():
            for line in ssh_config.read_text().splitlines():
                if line.strip().startswith("Port "):
                    port = int(line.strip().split()[1])
                    break
        return {
            "host": ip,
            "port": port,
            "user": "root",
            "ssh_config": str(ssh_config) if ssh_config.exists() else None,
            "ssh_command": f"ssh -F {ssh_config} {cluster_name}" if ssh_config.exists() else f"ssh root@{ip}",
        }

    def submit(self, cluster_name: str, command: str) -> int:
        """
        Submit a job to a cluster (queued, non-blocking).
        Returns job ID.
        """
        result = subprocess.run(
            [_sky(), "exec", cluster_name, "--", command],
            capture_output=True, text=True
        )
        return result.returncode

    def logs(self, cluster_name: str, job_id: str = "1") -> None:
        """Stream logs from a cluster job."""
        subprocess.run([_sky(), "logs", cluster_name, job_id])

    def start(self, cluster_name: str) -> None:
        """Restart a stopped cluster."""
        subprocess.run([_sky(), "start", cluster_name, "--yes"], check=True)

    def stop(self, cluster_name: str) -> None:
        """Stop cluster (pauses billing, preserves state for resume)."""
        subprocess.run([_sky(), "stop", cluster_name, "--yes"], check=True)

    def down(self, cluster_name: str) -> None:
        """Terminate cluster permanently."""
        subprocess.run([_sky(), "down", cluster_name, "--yes"], check=True)

    def down_all(self) -> None:
        """Terminate all clusters."""
        subprocess.run([_sky(), "down", "--all", "--yes"], check=True)

    def status(self) -> str:
        """Get status of all clusters."""
        result = subprocess.run([_sky(), "status"], capture_output=True, text=True)
        return result.stdout

    def check(self) -> None:
        """Run sky check to verify cloud credentials."""
        subprocess.run([_sky(), "check"])

    def wait_for_ready(self, cluster_name: str, timeout: int = 600) -> bool:
        """
        Poll sky status until cluster is READY or timeout.
        Returns True if ready, False if timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            result = subprocess.run(
                [_sky(), "status", cluster_name],
                capture_output=True, text=True
            )
            if "READY" in result.stdout:
                return True
            if "FAILED" in result.stdout or "ERROR" in result.stdout:
                raise RuntimeError(f"Cluster {cluster_name} failed to start")
            time.sleep(10)
        return False

    def autostop(self, cluster_name: str, idle_minutes: int) -> None:
        """Set idle autostop on a running cluster."""
        subprocess.run([_sky(), "autostop", cluster_name, "-i", str(idle_minutes), "--yes"])
