import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analyzer import CodebaseAnalyzer
from .config import EnvironmentSpec
from .skypilot import SkyPilotLauncher, _sky
from .snapshot import EnvironmentSnapshotter
from .validator import EnvironmentValidator

console = Console()

ENV_FILE = Path.home() / ".gpu-lol" / ".env"
KEYS_FILE = Path.home() / ".gpu-lol" / ".env.keys"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    gpu-lol: AI-powered GPU environment manager

    Drop-in replacement for `sky` — with automatic environment detection.
    """
    pass


# ---------------------------------------------------------------------------
# up — analyze + launch (the main gpu-lol superpower)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("repo_path", default=".", metavar="[REPO_PATH]")
@click.option("--name", "-n", default=None, help="Cluster name (auto-generated if not provided)")
@click.option("--gpu", default=None, help="Force GPU type e.g. A40, H100-SXM, RTX4090")
@click.option("--provider", default=None, type=click.Choice(["runpod", "vast", "lambda"]),
              help="Force specific cloud provider")
@click.option("--dry-run", is_flag=True, help="Show detected environment + SkyPilot YAML without launching")
@click.option("--no-validate", is_flag=True, help="Skip environment validation after launch")
@click.option("--detach", is_flag=True, help="Return immediately after submitting — don't wait for setup")
@click.option("--template", default=None, metavar="NAME_OR_ID", help="Use a specific RunPod template (name or id). Run 'gpu-lol templates' to list.")
@click.option("--gpus", default=None, type=int, help="Number of GPUs (auto-detected from code if not set)")
@click.option("--assets", multiple=True, metavar="REMOTE:LOCAL",
              help="Symlink path on pod. E.g. --assets /workspace/models:/root/.cache/huggingface/hub")
@click.option("--yes", "-y", is_flag=True, help="Skip cost confirmation prompt")
@click.option("--stop-after", "autostop_hours", default=None, type=float,
              metavar="HOURS", help="Auto-stop cluster after N hours of idle (e.g. --stop-after 2)")
@click.option("--watch", is_flag=True, help="With --detach: tail the launch log after firing (Ctrl-C safe)")
def up(repo_path, name, gpu, provider, dry_run, no_validate, detach, template, gpus, assets, yes, autostop_hours, watch):
    """
    Analyze repo and spin up optimal GPU environment.

    Examples:\n
      gpu-lol up                     # Analyze current directory\n
      gpu-lol up ~/my-llm-project    # Analyze specific repo\n
      gpu-lol up --gpu H100-SXM      # Force specific GPU\n
      gpu-lol up --dry-run           # Preview without launching
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.exists():
        console.print(f"[red]✗ Path not found: {repo_path}[/red]")
        raise click.Abort()

    # First-run: redirect to secrets init if no credentials configured
    if not ENV_FILE.exists():
        console.print("\n[yellow]⚠ No credentials found.[/yellow]")
        console.print("  Looks like your first time — run setup first:\n")
        console.print("  [bold]gpu-lol secrets init[/bold]\n")
        raise click.Abort()

    cluster_name = name or f"gr-{repo_path.name}-{int(time.time()) % 100000}"

    # Step 1: Analyze
    console.print(f"\n[bold blue]🔍 Analyzing {repo_path.name}...[/bold blue]")
    analyzer = CodebaseAnalyzer(str(repo_path))
    spec = analyzer.analyze(template_override=template)

    # Inject HF_TOKEN into pod environment if configured
    from .llm import _load_config
    _cfg = _load_config()
    hf_token_val = _cfg.get("HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")
    if hf_token_val:
        # Write token to pod during setup
        spec.setup_commands = [f"mkdir -p ~/.cache/huggingface && echo '{hf_token_val}' > ~/.cache/huggingface/token"] + spec.setup_commands

    # Asset symlinks: user-specified + auto-detected HF cache for HF projects
    all_assets = list(assets)
    if not all_assets:
        packages_lower = [p.lower() for p in spec.requirements]
        uses_hf = any(p in packages_lower for p in
                      ["transformers", "diffusers", "datasets", "huggingface-hub", "peft", "trl"])
        if uses_hf:
            all_assets = ["/workspace/huggingface:/root/.cache/huggingface"]
            console.print(f"  [dim]Auto-asset: /workspace/huggingface → ~/.cache/huggingface (RunPod network vol)[/dim]")
    if all_assets:
        spec.asset_links = all_assets

    if gpu:
        spec.gpu_type = gpu
        console.print(f"  [dim]GPU overridden to: {gpu}[/dim]")

    if gpus:
        spec.gpu_count = gpus
        console.print(f"  [dim]GPUs overridden to: {gpus}[/dim]")

    console.print(f"[green]✓[/green] Environment detected:")
    console.print(f"  Workload:  [bold]{spec.workload_type}[/bold]")
    console.print(f"  GPU:       [bold]{spec.gpu_count}x {spec.gpu_type}[/bold] (needs {spec.vram_required_gb}GB VRAM)")
    if autostop_hours:
        console.print(f"  Auto-stop: [bold]{autostop_hours}h[/bold] idle timeout")
    console.print(f"  Image:     [dim]{spec.base_image}[/dim]")
    console.print(f"  Packages:  {len(spec.requirements)} dependencies")
    if spec.asset_links:
        console.print(f"  Assets:    {len(spec.asset_links)} symlink(s) configured")

    # Warn if no ML packages detected — user might be in the wrong directory
    ML_PACKAGES = {"torch", "tensorflow", "transformers", "diffusers", "jax",
                   "vllm", "unsloth", "peft", "trl", "accelerate", "keras"}
    detected_ml = ML_PACKAGES & {p.split("==")[0].split(">=")[0].split("<=")[0].lower()
                                  for p in spec.requirements}
    if not detected_ml and not dry_run:
        console.print(f"\n[yellow]⚠ No ML packages detected in {repo_path.name}.[/yellow]")
        console.print(f"  Are you in the right directory?")
        if not click.confirm("  Launch anyway?", default=False):
            raise click.Abort()

    launcher = SkyPilotLauncher()

    if dry_run:
        console.print(f"\n[bold]Generated SkyPilot YAML:[/bold]")
        yaml_content = launcher.generate_task_yaml(spec, str(repo_path))
        console.print(Panel(yaml_content, title="task.yaml", border_style="dim"))
        return

    # Step 2: Launch
    console.print(f"\n[bold blue]🚀 Launching cluster '{cluster_name}'...[/bold blue]")
    console.print(f"  [dim]SkyPilot will find cheapest available {spec.gpu_type} across RunPod, Vast.ai, Lambda[/dim]\n")

    # Show cost estimate and confirm
    gpu_costs = {
        "RTX3090": 0.22, "RTX4090": 0.34, "A40": 0.40,
        "A6000": 0.50, "A100-SXM4": 1.19, "H100-SXM": 2.49,
    }
    cost_hr = gpu_costs.get(spec.gpu_type, 0.0) * spec.gpu_count
    if cost_hr > 0:
        console.print(f"  [bold]Estimated cost:[/bold] ~${cost_hr:.2f}/hr", end="")
        if autostop_hours:
            console.print(f"  ([dim]auto-stops after {autostop_hours}h → max ~${cost_hr * autostop_hours:.2f}[/dim])")
        else:
            console.print("")
    if not yes and not dry_run:
        if not click.confirm(f"\n  Launch {spec.gpu_count}x {spec.gpu_type} @ ~${cost_hr:.2f}/hr?", default=True):
            raise click.Abort()

    try:
        launcher.launch(spec, cluster_name, str(repo_path), detach=detach, autostop_hours=autostop_hours)
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]✗ Launch failed.[/red]")

        # Parse output for known error patterns
        err_output = getattr(e, 'output', '') or ''

        if "ResourcesUnavailableError" in err_output or "Failed to acquire resources" in err_output:
            console.print(f"\n[yellow]⚠ No {spec.gpu_type} available right now.[/yellow]")
            # Suggest the next GPU up
            _suggest_gpu_alternatives(console, spec.gpu_type)
        elif "Authentication" in err_output or "api_key" in err_output.lower():
            console.print(f"  [dim]Credentials issue — run: gpu-lol check[/dim]")
            console.print(f"  [dim]Then: gpu-lol secrets show[/dim]")
        else:
            console.print(f"  Try: [bold]gpu-lol up --dry-run[/bold]  (validate YAML)")
            console.print(f"  Try: [bold]gpu-lol check[/bold]         (verify credentials)")
        raise click.Abort()

    if detach:
        console.print(Panel(
            Text.from_markup(
                f"[bold green]🚀 Launched![/bold green] Provisioning in background.\n\n"
                f"[bold]Check status:[/bold]  gpu-lol ls\n"
                f"[bold]Connect:[/bold]       gpu-lol ssh {cluster_name}\n"
                f"[bold]Validate:[/bold]      gpu-lol validate {cluster_name}\n"
                f"[bold]Logs:[/bold]          gpu-lol logs {cluster_name}\n\n"
                f"[bold]Watch launch:[/bold]  tail -f /tmp/gpu-lol-{cluster_name}.log\n"
                f"[dim]Cluster will be ready in ~2-5 minutes[/dim]"
            ),
            border_style="blue"
        ))
        if watch:
            log_path = Path(tempfile.gettempdir()) / f"gpu-lol-{cluster_name}.log"
            console.print(f"[dim]Watching launch log (Ctrl-C to stop watching, cluster keeps running)...[/dim]\n")
            try:
                subprocess.run(["tail", "-f", str(log_path)])
            except KeyboardInterrupt:
                console.print(f"\n[dim]Stopped watching. Cluster is still running.[/dim]")
                console.print(f"  gpu-lol ls\n  gpu-lol ssh {cluster_name}")
        return

    # Step 3: Wait for ready
    console.print(f"\n[bold blue]⏳ Waiting for cluster to be ready...[/bold blue]")
    ready = launcher.wait_for_ready(cluster_name)
    if not ready:
        console.print(f"[yellow]⚠ Cluster did not reach READY state within timeout[/yellow]")
        console.print(f"  Check: gpu-lol ls")
        raise click.Abort()

    # Step 4: Validate
    if not no_validate:
        _run_validate(launcher, cluster_name, spec)

    # Step 5: Hand off
    _print_ssh_panel(launcher, cluster_name)


# ---------------------------------------------------------------------------
# validate — run checks on a running cluster
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("cluster_name")
def validate(cluster_name):
    """
    Validate a running cluster's GPU, CUDA, and packages.

    Examples:\n
      gpu-lol validate gr-mock-test
    """
    console.print(f"\n[bold blue]🧪 Validating {cluster_name}...[/bold blue]")
    launcher = SkyPilotLauncher()

    # Load spec from .gpu-lol.yaml in current dir if it exists, else use defaults
    spec_path = Path(".gpu-lol.yaml")
    spec = EnvironmentSpec.from_yaml_file(str(spec_path)) if spec_path.exists() else EnvironmentSpec()

    _run_validate(launcher, cluster_name, spec)
    _print_ssh_panel(launcher, cluster_name)


# ---------------------------------------------------------------------------
# ssh — drop into a cluster shell
# ---------------------------------------------------------------------------

@cli.command(name="ssh")
@click.argument("cluster_name")
@click.argument("cmd", nargs=-1)
def ssh_cmd(cluster_name, cmd):
    """
    SSH into a running cluster (or run a command on it).

    Examples:\n
      gpu-lol ssh gr-mock-test\n
      gpu-lol ssh gr-mock-test -- nvidia-smi
    """
    # sky ssh was redesigned in 0.11 for node pools — use sky's generated SSH config
    launcher = SkyPilotLauncher()
    info = launcher.get_ssh_info(cluster_name)
    if not info.get("host"):
        console.print(f"[red]✗ Could not resolve IP for '{cluster_name}'. Is it running?[/red]")
        console.print("  Try: gpu-lol ls")
        raise click.Abort()
    if info.get("ssh_config"):
        args = ["ssh", "-F", info["ssh_config"], cluster_name]
    else:
        args = ["ssh", "-o", "StrictHostKeyChecking=no", f"root@{info['host']}"]
    if cmd:
        args += list(cmd)
    subprocess.run(args)


# ---------------------------------------------------------------------------
# exec — submit a job to a cluster (queued, streamed)
# ---------------------------------------------------------------------------

@cli.command(name="exec")
@click.argument("cluster_name")
@click.argument("cmd", nargs=-1, required=True)
def exec_cmd(cluster_name, cmd):
    """
    Submit a command to a running cluster (queued job).

    Examples:\n
      gpu-lol exec gr-mock-test -- python train.py\n
      gpu-lol exec gr-mock-test -- nvidia-smi
    """
    subprocess.run([_sky(), "exec", cluster_name, "--"] + list(cmd))


# ---------------------------------------------------------------------------
# logs — stream cluster logs
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("cluster_name")
@click.argument("job_id", default="1")
def logs(cluster_name, job_id):
    """
    Stream logs from a cluster job.

    Examples:\n
      gpu-lol logs gr-mock-test\n
      gpu-lol logs gr-mock-test 2
    """
    subprocess.run([_sky(), "logs", cluster_name, job_id])


# ---------------------------------------------------------------------------
# snapshot — capture running environment to .gpu-lol.yaml
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("cluster_name")
@click.option("--save", "-s", default=".", metavar="PATH",
              help="Directory to save .gpu-lol.yaml (default: current directory)")
def snapshot(cluster_name, save):
    """
    Snapshot running environment to .gpu-lol.yaml

    Examples:\n
      gpu-lol snapshot my-cluster\n
      gpu-lol snapshot my-cluster --save ~/my-project
    """
    console.print(f"\n[bold blue]📸 Snapshotting {cluster_name}...[/bold blue]")
    launcher = SkyPilotLauncher()
    snapshotter = EnvironmentSnapshotter(launcher, cluster_name)

    with console.status("Capturing environment state..."):
        spec = snapshotter.snapshot()

    snapshotter.save(spec, save)


# ---------------------------------------------------------------------------
# resume — relaunch from .gpu-lol.yaml
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("spec_path", default=".gpu-lol.yaml", metavar="[SPEC_PATH]")
@click.option("--name", "-n", default=None, help="Cluster name")
@click.option("--dry-run", is_flag=True)
def resume(spec_path, name, dry_run):
    """
    Resume environment from .gpu-lol.yaml snapshot.

    Examples:\n
      gpu-lol resume\n
      gpu-lol resume ~/project/.gpu-lol.yaml
    """
    if not Path(spec_path).exists():
        console.print(f"[red]✗ Spec file not found: {spec_path}[/red]")
        console.print("  Run 'gpu-lol snapshot <cluster>' first to create one.")
        raise click.Abort()

    spec = EnvironmentSpec.from_yaml_file(spec_path)
    cluster_name = name or f"gr-resume-{int(time.time()) % 100000}"

    console.print(f"\n[bold blue]🔄 Resuming from {spec_path}...[/bold blue]")
    console.print(f"  GPU: [bold]{spec.gpu_type}[/bold] ({spec.vram_required_gb}GB VRAM)")
    console.print(f"  Packages: {len(spec.requirements)}")

    launcher = SkyPilotLauncher()

    if dry_run:
        yaml_content = launcher.generate_task_yaml(spec, ".")
        console.print(Panel(yaml_content, title="Generated task.yaml"))
        return

    console.print(f"\n[bold blue]🚀 Launching '{cluster_name}'...[/bold blue]\n")
    launcher.launch(spec, cluster_name, ".")
    ready = launcher.wait_for_ready(cluster_name)
    if ready:
        _print_ssh_panel(launcher, cluster_name)


# ---------------------------------------------------------------------------
# ls / status — list clusters
# ---------------------------------------------------------------------------

@cli.command(name="ls")
def list_clusters():
    """List all running GPU clusters."""
    subprocess.run([_sky(), "status"])


@cli.command()
def status():
    """Show full SkyPilot status (clusters, jobs, services)."""
    subprocess.run([_sky(), "status"])


@cli.command(name="info")
@click.argument("cluster_name")
def cluster_info(cluster_name):
    """
    Show detailed info for a single cluster: GPU, IP, cost, SSH command.

    Examples:\n
      gpu-lol info gr-my-project
    """
    result = subprocess.run(
        [_sky(), "status", "--all", cluster_name],
        capture_output=True, text=True
    )
    raw = result.stdout + result.stderr

    if "No existing clusters" in raw or cluster_name not in raw:
        console.print(f"[red]✗ Cluster '{cluster_name}' not found.[/red]")
        console.print("  gpu-lol ls")
        raise click.Abort()

    # Get IP
    launcher = SkyPilotLauncher()
    info = launcher.get_ssh_info(cluster_name)

    # Parse GPU/status from sky status output
    gpu_type = "unknown"
    status_str = "unknown"
    cost_hr = 0.0
    for line in raw.splitlines():
        if cluster_name in line:
            parts = line.split()
            for p in parts:
                if any(g in p for g in ["RTX", "A40", "A100", "H100", "A6000"]):
                    gpu_type = p.strip(",")
            if "UP" in parts:
                status_str = "UP"
            elif "STOPPED" in parts:
                status_str = "STOPPED"

    gpu_costs = {
        "RTX3090": 0.22, "RTX4090": 0.34, "A40": 0.40,
        "A6000": 0.50, "A100-SXM4": 1.19, "H100-SXM": 2.49,
    }
    for key in gpu_costs:
        if key in gpu_type:
            cost_hr = gpu_costs[key]
            break

    status_color = "green" if status_str == "UP" else "yellow"
    console.print(Panel(
        Text.from_markup(
            f"[bold]{cluster_name}[/bold]\n\n"
            f"[bold]Status:[/bold]   [{status_color}]{status_str}[/{status_color}]\n"
            f"[bold]GPU:[/bold]      {gpu_type}\n"
            f"[bold]Cost:[/bold]     ~${cost_hr:.2f}/hr\n"
            f"[bold]IP:[/bold]       {info.get('host', 'not available')}\n"
            f"[bold]Port:[/bold]     {info.get('port', 22)}\n\n"
            f"[bold]SSH:[/bold]      {info.get('ssh_command', 'not available')}\n\n"
            f"[dim]gpu-lol logs {cluster_name}[/dim]\n"
            f"[dim]gpu-lol stop {cluster_name}[/dim]   pause billing\n"
            f"[dim]gpu-lol down {cluster_name}[/dim]   terminate"
        ),
        border_style="blue",
        title=f"Cluster Info"
    ))


# ---------------------------------------------------------------------------
# start — restart a stopped cluster
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("cluster_name")
def start(cluster_name):
    """
    Restart a stopped cluster (resumes billing).

    Examples:\n
      gpu-lol start gr-mock-test
    """
    with console.status(f"Starting {cluster_name}..."):
        SkyPilotLauncher().start(cluster_name)
    console.print(f"[green]✓[/green] Started {cluster_name}")
    _print_ssh_panel(SkyPilotLauncher(), cluster_name)


# ---------------------------------------------------------------------------
# stop — pause a cluster
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("cluster_name")
def stop(cluster_name):
    """
    Stop a cluster (pauses billing, state preserved).

    Resume with: gpu-lol start <cluster_name>
    """
    with console.status(f"Stopping {cluster_name}..."):
        SkyPilotLauncher().stop(cluster_name)
    console.print(f"[green]✓[/green] Stopped {cluster_name}  [dim](billing paused)[/dim]")


# ---------------------------------------------------------------------------
# down — terminate permanently
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("cluster_name", default="")
@click.option("--all", "all_clusters", is_flag=True, help="Terminate all clusters")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def down(cluster_name, all_clusters, yes):
    """
    Terminate cluster(s) permanently.

    Examples:\n
      gpu-lol down gr-mock-test\n
      gpu-lol down --all
    """
    if all_clusters:
        # Show what's running before confirming
        result = subprocess.run([_sky(), "status"], capture_output=True, text=True)
        raw = result.stdout
        running = [line for line in raw.splitlines()
                   if any(s in line for s in ["UP", "STOPPED"]) and "NAME" not in line
                   and line.strip() and not line.startswith("Managed") and not line.startswith("No")]

        if not running:
            console.print("[dim]No clusters running.[/dim]")
            return

        console.print(f"\n[bold yellow]⚠ About to terminate {len(running)} cluster(s):[/bold yellow]")
        for r in running:
            console.print(f"  [dim]{r.strip()}[/dim]")

        # Estimate total cost
        gpu_costs = {"RTX3090": 0.22, "RTX4090": 0.34, "A40": 0.40,
                     "A6000": 0.50, "A100-SXM4": 1.19, "H100-SXM": 2.49}
        total_cost = sum(cost for key, cost in gpu_costs.items()
                        if any(key in r for r in running))
        if total_cost > 0:
            console.print(f"  [dim]Stopping ~${total_cost:.2f}/hr in billing[/dim]")

        console.print("")
        if not yes:
            # Require explicit confirmation for --all even with no --yes flag
            confirm_text = click.prompt(
                '  Type "yes" to terminate all clusters',
                default=""
            )
            if confirm_text.lower() != "yes":
                console.print("[dim]Aborted.[/dim]")
                raise click.Abort()

        with console.status("Terminating all clusters..."):
            SkyPilotLauncher().down_all()
        console.print(f"[green]✓[/green] All clusters terminated")
        return

    if not cluster_name:
        console.print("[red]✗ Provide a cluster name or use --all[/red]")
        raise click.Abort()

    if not yes:
        click.confirm(
            f"Permanently terminate '{cluster_name}'?",
            abort=True
        )
    with console.status(f"Terminating {cluster_name}..."):
        SkyPilotLauncher().down(cluster_name)
    console.print(f"[green]✓[/green] Terminated {cluster_name}")


# ---------------------------------------------------------------------------
# check — verify cloud credentials
# ---------------------------------------------------------------------------

@cli.command()
def check():
    """Check which cloud providers are enabled and ready."""
    console.print("\n[bold blue]☁ Checking cloud providers...[/bold blue]\n")

    result = subprocess.run([_sky(), "check"], capture_output=True, text=True)
    raw = result.stdout + result.stderr

    # Parse sky check output into provider status
    table = Table(box=None, padding=(0, 2))
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Notes", style="dim")

    provider_map = {
        "runpod": ("RunPod", "RTX3090/4090, A40, A100, H100"),
        "vast":   ("Vast.ai", "Consumer GPUs, often cheaper"),
        "lambda": ("Lambda Labs", "Good A100 availability"),
        "aws":    ("AWS", "Enterprise"),
        "gcp":    ("GCP", "Enterprise"),
        "azure":  ("Azure", "Enterprise"),
    }

    enabled = []
    disabled = []

    for line in raw.splitlines():
        line_lower = line.lower()
        for key, (name, note) in provider_map.items():
            if key in line_lower:
                if "enabled" in line_lower and key in line_lower:
                    enabled.append((name, note))
                elif "disabled" in line_lower and key in line_lower:
                    disabled.append((name, note))

    for name, note in enabled:
        table.add_row(f"[green]✓[/green] {name}", "[green]enabled[/green]", note)
    for name, note in disabled:
        table.add_row(f"[dim]✗ {name}[/dim]", "[dim]disabled[/dim]", note)

    if not enabled and not disabled:
        # sky check output format didn't match — show raw
        console.print(raw)
        return

    console.print(table)

    if not enabled:
        console.print("\n[red]✗ No providers enabled.[/red]")
        console.print("  Run [bold]gpu-lol secrets init[/bold] to configure credentials.\n")
    elif not any(p[0] in ("RunPod", "Vast.ai", "Lambda Labs") for p in enabled):
        console.print("\n[yellow]⚠ No GPU cloud providers enabled.[/yellow]")
        console.print("  Run [bold]gpu-lol secrets init[/bold] to add RunPod, Vast.ai, or Lambda.\n")
    else:
        console.print(f"\n[green]✓ {len(enabled)} provider(s) ready.[/green]  gpu-lol up is good to go.\n")


# ---------------------------------------------------------------------------
# templates — list and select RunPod pod templates
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--filter", "-f", "filter_str", default=None, help="Filter by name (partial match)")
def templates(filter_str):
    """
    List your RunPod pod templates. These boot fast — images are cached on RunPod nodes.

    Examples:\n
      gpu-lol templates\n
      gpu-lol templates --filter porpoise\n
      gpu-lol up . --template competitive_salmon_porpoise
    """
    from . import runpod_api

    console.print("\n[bold blue]📦 Fetching RunPod templates...[/bold blue]")
    all_templates = runpod_api.fetch_templates()

    if not all_templates:
        console.print("[red]✗ Could not fetch templates. Check your RunPod API key:[/red]")
        console.print("  gpu-lol secrets show")
        return

    if filter_str:
        all_templates = [t for t in all_templates if filter_str.lower() in t["name"].lower()
                         or filter_str.lower() in t["imageName"].lower()]

    # Highlight user's personal templates (fast-boot)
    personal = [t for t in all_templates if "competitive_salmon_porpoise" in t["name"].lower()]
    official = [t for t in all_templates if t.get("id", "").startswith("runpod-torch")]
    other = [t for t in all_templates if t not in personal and t not in official]

    def _print_group(title: str, items: list, highlight: bool = False):
        if not items:
            return
        console.print(f"\n[bold]{title}[/bold]")
        table = Table(box=None, padding=(0, 2))
        table.add_column("Name", style="bold" if highlight else "")
        table.add_column("Image")
        table.add_column("ID", style="dim")
        for t in items:
            name = f"[green]{t['name']}[/green]" if highlight else t["name"]
            table.add_row(name, t["imageName"], t["id"])
        console.print(table)

    _print_group("⚡ Your templates (fast boot — cached on RunPod nodes)", personal, highlight=True)
    _print_group("🐳 Official RunPod templates", official)
    _print_group("📦 Other templates", other)

    # Show which one gpu-lol would auto-select
    auto = runpod_api.best_ml_template("training")
    if auto:
        console.print(f"\n[dim]Auto-selected for ML workloads: [bold]{auto['name']}[/bold][/dim]")

    console.print(f"\n[dim]Use with: gpu-lol up . --template <name or id>[/dim]\n")


# ---------------------------------------------------------------------------
# secrets — credential management
# ---------------------------------------------------------------------------

@cli.group()
def secrets():
    """Manage credentials (RunPod key, LLM config). Encrypted at rest with dotenvx."""
    pass


@secrets.command(name="init")
def secrets_init():
    """
    Interactive setup wizard — configure all credentials and encrypt them.

    Examples:\n
      gpu-lol secrets init
    """
    console.print("\n[bold blue]🔑 gpu-lol credential setup[/bold blue]\n")
    console.print("[dim]You'll need:[/dim]")
    console.print("  [dim]• RunPod account with payment method: https://www.runpod.io[/dim]")
    console.print("  [dim]• RunPod API key: https://www.runpod.io/console/user/settings[/dim]")
    console.print("  [dim]• (Optional) Vast.ai or Lambda account for more availability[/dim]\n")

    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing values so we can show them as defaults
    existing = _read_env_plain()

    # RunPod
    console.print("[bold]RunPod[/bold]  [dim]https://www.runpod.io/console/user/settings[/dim]")
    current_rp = existing.get("RUNPOD_API_KEY", "")
    if current_rp:
        console.print(f"  Current: [dim]{'*' * 8}{current_rp[-4:]}[/dim]")
    runpod_key = click.prompt("  API key", default=current_rp or "", show_default=False, hide_input=True)
    if not runpod_key:
        console.print("[red]✗ RunPod API key is required[/red]")
        raise click.Abort()

    # Vast.ai (optional)
    console.print("\n[bold]Vast.ai[/bold]  [dim]optional — https://vast.ai/account[/dim]")
    vast_key_file = Path.home() / ".vast_api_key"
    current_vast = vast_key_file.read_text().strip() if vast_key_file.exists() else existing.get("VAST_API_KEY", "")
    if current_vast:
        console.print(f"  Current: [dim]{'*' * 8}{current_vast[-4:]}[/dim]")
    vast_key = click.prompt("  API key (Enter to skip)", default="", show_default=False, hide_input=True)

    # Lambda Cloud (optional)
    console.print("\n[bold]Lambda Cloud[/bold]  [dim]optional — https://cloud.lambdalabs.com/api-keys[/dim]")
    lambda_config_file = Path.home() / ".lambda_cloud" / "lambda_keys.yaml"
    current_lambda = ""
    if lambda_config_file.exists():
        try:
            import yaml as _yaml
            ldata = _yaml.safe_load(lambda_config_file.read_text()) or {}
            current_lambda = ldata.get("api_key", "")
        except Exception:
            pass
    if current_lambda:
        console.print(f"  Current: [dim]{'*' * 8}{current_lambda[-4:]}[/dim]")
    lambda_key = click.prompt("  API key (Enter to skip)", default="", show_default=False, hide_input=True)

    # Template pattern (optional)
    console.print("\n[bold]Template pattern[/bold]  [dim]optional — your RunPod template name prefix[/dim]")
    current_pattern = existing.get("GPU_LOL_TEMPLATE_PATTERN", "")
    template_pattern = click.prompt(
        "  Pattern",
        default=current_pattern or "competitive_salmon_porpoise",
    )

    # LLM (optional)
    console.print("\n[bold]LLM[/bold]  [dim]optional — makes analysis smarter[/dim]")
    console.print("  [dim]Any OpenAI-compatible endpoint: OpenRouter, Anthropic, Ollama, vLLM[/dim]")
    current_url = existing.get("GPU_LOL_LLM_URL", "")
    llm_url = click.prompt("  LLM URL", default=current_url or "", show_default=bool(current_url))
    llm_key, llm_model = "", ""
    if llm_url:
        current_key = existing.get("GPU_LOL_LLM_KEY", "")
        if current_key:
            console.print(f"  Current key: [dim]{'*' * 8}{current_key[-4:]}[/dim]")
        llm_key = click.prompt("  API key", default=current_key or "", show_default=False, hide_input=True)
        llm_model = click.prompt(
            "  Model",
            default=existing.get("GPU_LOL_LLM_MODEL", "x-ai/grok-4.1-fast"),
        )

    # HuggingFace token (optional — needed for gated models like Llama, Gemma)
    console.print("\n[bold]HuggingFace[/bold]  [dim]optional — needed for gated models (Llama, Gemma, etc.)[/dim]")
    console.print("  [dim]Get token: https://huggingface.co/settings/tokens[/dim]")
    hf_token_file = Path.home() / ".cache" / "huggingface" / "token"
    current_hf = existing.get("HF_TOKEN", "")
    if not current_hf and hf_token_file.exists():
        current_hf = hf_token_file.read_text().strip()
    if current_hf:
        console.print(f"  Current: [dim]{'*' * 8}{current_hf[-4:]}[/dim]")
    hf_token = click.prompt("  Token (Enter to skip)", default="", show_default=False, hide_input=True)

    # Write plain .env
    content = (
        f"RUNPOD_API_KEY={runpod_key}\n"
        f"GPU_LOL_LLM_URL={llm_url}\n"
        f"GPU_LOL_LLM_KEY={llm_key}\n"
        f"GPU_LOL_LLM_MODEL={llm_model}\n"
        f"GPU_LOL_TEMPLATE_PATTERN={template_pattern}\n"
        f"HF_TOKEN={hf_token}\n"
    )
    ENV_FILE.write_text(content)
    ENV_FILE.chmod(0o600)

    # Encrypt
    console.print("")
    dotenvx = _dotenvx_bin()
    if dotenvx:
        result = subprocess.run(
            [dotenvx, "encrypt", "-f", str(ENV_FILE)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            KEYS_FILE.chmod(0o600)
            console.print(f"[green]✓[/green] Credentials encrypted  [dim](~/.gpu-lol/.env)[/dim]")
            console.print(f"[green]✓[/green] Private key saved       [dim](~/.gpu-lol/.env.keys — back this up)[/dim]")
        else:
            console.print(f"[yellow]⚠[/yellow] Encryption failed — stored as plain text (chmod 600)")
    else:
        console.print(f"[green]✓[/green] Credentials saved  [dim](~/.gpu-lol/.env, chmod 600)[/dim]")
        console.print(f"[dim]  Install dotenvx for encryption: npm install -g @dotenvx/dotenvx[/dim]")

    # Configure SkyPilot RunPod integration
    runpod_config = Path.home() / ".runpod" / "config.toml"
    runpod_config.parent.mkdir(parents=True, exist_ok=True)
    runpod_config.write_text(f"[default]\napi_key = \"{runpod_key}\"\n")
    runpod_config.chmod(0o600)
    console.print(f"[green]✓[/green] RunPod configured      [dim](~/.runpod/config.toml)[/dim]")

    # Configure Vast.ai
    if vast_key:
        vast_key_file.write_text(vast_key)
        vast_key_file.chmod(0o600)
        console.print(f"[green]✓[/green] Vast.ai configured      [dim](~/.vast_api_key)[/dim]")

    # Configure Lambda Cloud
    if lambda_key:
        lambda_config_file.parent.mkdir(parents=True, exist_ok=True)
        import yaml as _yaml
        lambda_config_file.write_text(_yaml.dump({"api_key": lambda_key}))
        lambda_config_file.chmod(0o600)
        console.print(f"[green]✓[/green] Lambda configured        [dim](~/.lambda_cloud/lambda_keys.yaml)[/dim]")

    # Configure HuggingFace token
    if hf_token:
        hf_token_file.parent.mkdir(parents=True, exist_ok=True)
        hf_token_file.write_text(hf_token)
        hf_token_file.chmod(0o600)
        console.print(f"[green]✓[/green] HuggingFace configured  [dim](~/.cache/huggingface/token)[/dim]")

    console.print("\nRun [bold]gpu-lol secrets show[/bold] to verify, [bold]gpu-lol check[/bold] to test providers.\n")


@secrets.command(name="show")
def secrets_show():
    """
    Show current credentials (keys masked).

    Examples:\n
      gpu-lol secrets show
    """
    config = _read_env_decrypted()
    if not config:
        console.print("[yellow]⚠ No credentials configured. Run: gpu-lol secrets init[/yellow]")
        return

    encrypted = _is_encrypted()
    status = "[green]encrypted (dotenvx)[/green]" if encrypted else "[yellow]plain text[/yellow]"

    table = Table(box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    for key, val in config.items():
        if key.startswith("DOTENV_"):
            continue
        if ("KEY" in key or "key" in key) and val:
            display = f"{'*' * 8}{val[-4:]}"
        else:
            display = val or "[dim](not set)[/dim]"
        table.add_row(key, display)

    console.print(f"\n[bold]Credentials[/bold]  {status}")
    console.print(table)
    console.print(f"\n[dim]~/.gpu-lol/.env[/dim]")
    if encrypted:
        console.print(f"[dim]~/.gpu-lol/.env.keys  (private key — back this up)[/dim]")
    console.print("")


@secrets.command(name="set")
@click.argument("assignment", metavar="KEY=VALUE")
def secrets_set(assignment):
    """
    Set or update a single credential.

    Examples:\n
      gpu-lol secrets set RUNPOD_API_KEY=rpa_newkey\n
      gpu-lol secrets set GPU_LOL_LLM_MODEL=claude-sonnet-4-6
    """
    if "=" not in assignment:
        console.print("[red]✗ Format: gpu-lol secrets set KEY=VALUE[/red]")
        raise click.Abort()

    key, _, value = assignment.partition("=")
    key = key.strip()

    dotenvx = _dotenvx_bin()
    if dotenvx and _is_encrypted():
        result = subprocess.run(
            [dotenvx, "set", "-f", str(ENV_FILE), f"{key}={value}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            console.print(f"[green]✓[/green] {key} updated (encrypted)")
        else:
            console.print(f"[red]✗ Failed:[/red] {result.stderr.strip()}")
            raise click.Abort()
    else:
        # Plain text update
        config = _read_env_plain()
        config[key] = value
        lines = "\n".join(f"{k}={v}" for k, v in config.items()) + "\n"
        ENV_FILE.write_text(lines)
        ENV_FILE.chmod(0o600)
        console.print(f"[green]✓[/green] {key} updated")

    # Keep RunPod config.toml in sync
    if key == "RUNPOD_API_KEY":
        runpod_config = Path.home() / ".runpod" / "config.toml"
        runpod_config.parent.mkdir(parents=True, exist_ok=True)
        runpod_config.write_text(f"[default]\napi_key = \"{value}\"\n")
        runpod_config.chmod(0o600)
        console.print(f"[green]✓[/green] ~/.runpod/config.toml updated")


@secrets.command(name="encrypt")
def secrets_encrypt():
    """
    Encrypt credentials with dotenvx (run once after secrets init).

    Examples:\n
      gpu-lol secrets encrypt
    """
    dotenvx = _dotenvx_bin()
    if not dotenvx:
        console.print("[red]✗ dotenvx not found.[/red]")
        console.print("  Install: [bold]npm install -g @dotenvx/dotenvx[/bold]")
        raise click.Abort()

    if not ENV_FILE.exists():
        console.print("[red]✗ No credentials found. Run: gpu-lol secrets init[/red]")
        raise click.Abort()

    if _is_encrypted():
        console.print("[green]✓ Already encrypted[/green]")
        return

    result = subprocess.run(
        [dotenvx, "encrypt", "-f", str(ENV_FILE)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        KEYS_FILE.chmod(0o600)
        console.print(f"[green]✓[/green] Encrypted  [dim]~/.gpu-lol/.env[/dim]")
        console.print(f"[green]✓[/green] Private key [dim]~/.gpu-lol/.env.keys[/dim]  ← back this up")
    else:
        console.print(f"[red]✗ Encryption failed:[/red] {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# secrets helpers
# ---------------------------------------------------------------------------

def _dotenvx_bin() -> str | None:
    """Find dotenvx binary."""
    return (
        shutil.which("dotenvx")
        or shutil.which("dotenvx", path=str(Path.home() / ".local" / "bin"))
        or (str(Path.home() / ".local" / "bin" / "dotenvx") if (Path.home() / ".local" / "bin" / "dotenvx").exists() else None)
    )


def _is_encrypted() -> bool:
    """Check if .env is dotenvx-encrypted."""
    if not ENV_FILE.exists():
        return False
    return "DOTENV_PUBLIC_KEY" in ENV_FILE.read_text()


def _read_env_plain() -> dict:
    """Read plain KEY=VALUE .env file."""
    if not ENV_FILE.exists():
        return {}
    config = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            config[k.strip()] = v.strip().strip('"').strip("'")
    return config


def _read_env_decrypted() -> dict:
    """Read .env, decrypting with dotenvx if encrypted."""
    dotenvx = _dotenvx_bin()
    if dotenvx and _is_encrypted():
        result = subprocess.run(
            [dotenvx, "get", "-f", str(ENV_FILE), "--format", "json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    return _read_env_plain()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suggest_gpu_alternatives(console: Console, gpu_type: str):
    """Suggest alternative GPUs when the requested one is sold out."""
    # Ordered catalog: each GPU's fallback options
    alternatives = {
        "RTX3090":   ["RTX4090", "A40"],
        "RTX4090":   ["A40", "A6000", "RTX3090"],
        "A40":       ["A6000", "A100-SXM4", "RTX4090"],
        "A6000":     ["A40", "A100-SXM4"],
        "A100-SXM4": ["H100-SXM", "A40"],
        "H100-SXM":  ["A100-SXM4"],
    }
    suggestions = alternatives.get(gpu_type, ["A40", "RTX4090"])
    console.print(f"\n  [bold]Try an alternative GPU:[/bold]")
    for alt in suggestions[:2]:
        console.print(f"    gpu-lol up --gpu {alt}")
    console.print(f"\n  [dim]Or retry in a few minutes — availability changes quickly.[/dim]")
    console.print(f"  [dim]See all providers: gpu-lol check[/dim]")


def _run_validate(launcher: SkyPilotLauncher, cluster_name: str, spec: EnvironmentSpec):
    console.print(f"\n[bold blue]🧪 Validating environment...[/bold blue]")
    validator = EnvironmentValidator(launcher, cluster_name)
    result = validator.validate(spec)

    if result.passed:
        console.print(f"[green]✓ {result.summary}[/green]")
    else:
        console.print(f"[yellow]⚠ {result.summary}[/yellow]")
        for f in result.failures:
            console.print(f"  [red]✗[/red] {f.name}: {f.error[:120]}")
        console.print("  Auto-fixing...")
        if validator.auto_fix(result.failures, spec):
            console.print("[green]✓ Fixed![/green]")
        else:
            console.print("[yellow]⚠ Some issues remain — environment may still work.[/yellow]")


def _print_ssh_panel(launcher: SkyPilotLauncher, cluster_name: str):
    try:
        ssh_info = launcher.get_ssh_info(cluster_name)
        console.print(Panel(
            Text.from_markup(
                f"[bold green]✨ Ready![/bold green]\n\n"
                f"[bold]SSH:[/bold]       {ssh_info['ssh_command']}\n\n"
                f"[bold]Your code:[/bold] ~/sky_workdir/\n\n"
                f"[dim]gpu-lol logs {cluster_name}[/dim]       stream logs\n"
                f"[dim]gpu-lol snapshot {cluster_name}[/dim]   save environment\n"
                f"[dim]gpu-lol stop {cluster_name}[/dim]       pause billing\n"
                f"[dim]gpu-lol down {cluster_name}[/dim]       terminate"
            ),
            border_style="green"
        ))
    except Exception:
        console.print(f"\n[green]✨ Ready![/green]  →  [bold]gpu-lol ssh {cluster_name}[/bold]")
