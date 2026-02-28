"""
Doctor Context MCP Server — Gives the AI "eyes" into your local dev environment.

Diagnoses "works on my machine" issues for Java, .NET, and Python projects.
Audits dependencies, traces port conflicts, fetches Docker logs, and finds local docs.
"""

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"

from mcp.server.fastmcp import FastMCP

_LOG_FILE = Path(__file__).resolve().parent.parent / "doctor-context-debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("doctor-context")

mcp = FastMCP("Doctor Context Server")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run(cmd: str | list[str], shell: bool = True, timeout: int = 10, cwd: str | None = None) -> dict:
    """Run a command and return stdout/stderr/returncode."""
    t0 = time.perf_counter()
    cmd_label = cmd if isinstance(cmd, str) else " ".join(cmd)
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=shell, cwd=cwd,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            elapsed = time.perf_counter() - t0
            logger.debug("_run [%.3fs] rc=%d cmd=%s", elapsed, proc.returncode, cmd_label[:120])
            return {
                "stdout": (stdout or "").strip(),
                "stderr": (stderr or "").strip(),
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            # On Windows, shell=True spawns cmd.exe; kill the entire process tree
            if _IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    capture_output=True, timeout=5,
                )
            else:
                proc.kill()
            proc.wait(timeout=5)
            elapsed = time.perf_counter() - t0
            logger.warning("_run [%.3fs] TIMEOUT cmd=%s", elapsed, cmd_label[:120])
            return {"stdout": "", "stderr": "Command timed out", "returncode": -1}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("_run [%.3fs] ERROR cmd=%s err=%s", elapsed, cmd_label[:120], e)
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def _tool_exists(name: str) -> bool:
    """Fast check whether a CLI tool is on PATH (no subprocess)."""
    return shutil.which(name) is not None


def _get_version(cmd: str) -> str:
    """Run a version command and return the output."""
    result = _run(cmd, timeout=5)
    if result["returncode"] == 0:
        output = result["stdout"] or result["stderr"]
        return output.split("\n")[0] if output else "installed (no version info)"
    return "NOT FOUND"


def _get_versions(cmds: dict[str, str]) -> dict[str, str]:
    """Run multiple version commands concurrently. cmds is {label: command}.
    Skips commands whose tool is not on PATH.
    """
    t0 = time.perf_counter()
    results = {}
    # Pre-filter: only run commands for tools that exist
    runnable = {}
    for label, cmd in cmds.items():
        tool_name = cmd.split()[0]
        if _tool_exists(tool_name):
            runnable[label] = cmd
        else:
            results[label] = "NOT FOUND"

    skipped = [l for l, v in results.items() if v == "NOT FOUND"]
    if skipped:
        logger.debug("_get_versions skipped (not on PATH): %s", skipped)

    if not runnable:
        return results

    with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
        futures = {pool.submit(_get_version, cmd): label for label, cmd in runnable.items()}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    elapsed = time.perf_counter() - t0
    logger.debug("_get_versions [%.3fs] checked: %s", elapsed, list(runnable.keys()))
    return results


def _detect_project_type(project_path: str) -> list[dict]:
    """Detect project types by looking for marker files."""
    path = Path(project_path)
    detected = []

    # Java
    if (path / "pom.xml").exists():
        detected.append({"type": "java", "build_tool": "maven", "marker": "pom.xml"})
    if (path / "build.gradle").exists() or (path / "build.gradle.kts").exists():
        marker = "build.gradle.kts" if (path / "build.gradle.kts").exists() else "build.gradle"
        detected.append({"type": "java", "build_tool": "gradle", "marker": marker})

    # Python
    if (path / "requirements.txt").exists():
        detected.append({"type": "python", "build_tool": "pip", "marker": "requirements.txt"})
    if (path / "pyproject.toml").exists():
        detected.append({"type": "python", "build_tool": "pyproject", "marker": "pyproject.toml"})
    if (path / "Pipfile").exists():
        detected.append({"type": "python", "build_tool": "pipenv", "marker": "Pipfile"})

    # .NET
    for csproj in path.glob("*.csproj"):
        detected.append({"type": "dotnet", "build_tool": "dotnet", "marker": csproj.name})
        break
    if (path / "*.sln").exists() or list(path.glob("*.sln")):
        for sln in path.glob("*.sln"):
            detected.append({"type": "dotnet", "build_tool": "dotnet-sln", "marker": sln.name})
            break

    return detected


# ═══════════════════════════════════════════════════════════════════════════
# Tool 1: audit_local_env
# ═══════════════════════════════════════════════════════════════════════════

def _audit_java(project_path: str) -> dict:
    """Audit Java environment against project requirements."""
    t0 = time.perf_counter()
    logger.info("_audit_java START for %s", project_path)
    path = Path(project_path)
    audit = {"runtime": {}, "build_tool": {}, "project_requirements": {}, "issues": []}

    # Only check build tools that the project actually uses
    cmds = {
        "java": "java -version 2>&1",
        "javac": "javac -version 2>&1",
    }
    has_maven = (path / "pom.xml").exists() or (path / "mvnw").exists() or (path / "mvnw.cmd").exists()
    has_gradle = (path / "build.gradle").exists() or (path / "build.gradle.kts").exists() or (path / "gradlew").exists()
    if has_maven:
        cmds["maven"] = "mvn -version 2>&1"
    if has_gradle:
        cmds["gradle"] = "gradle -version 2>&1"

    versions = _get_versions(cmds)
    audit["runtime"]["java"] = versions.get("java", "NOT FOUND")
    audit["runtime"]["javac"] = versions.get("javac", "NOT FOUND")
    if has_maven:
        audit["build_tool"]["maven"] = versions.get("maven", "NOT FOUND")
    if has_gradle:
        audit["build_tool"]["gradle"] = versions.get("gradle", "NOT FOUND")
    java_ver = versions.get("java", "")

    # Check JAVA_HOME
    java_home = os.environ.get("JAVA_HOME", "NOT SET")
    audit["runtime"]["JAVA_HOME"] = java_home
    if java_home == "NOT SET":
        audit["issues"].append("JAVA_HOME is not set — many build tools require this")

    # Parse pom.xml for Java version
    pom_path = path / "pom.xml"
    if pom_path.exists():
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}

            # Check maven.compiler.source/target in properties
            props = root.find(".//m:properties", ns) or root.find(".//properties")
            if props is not None:
                for child in props:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if "java" in tag.lower() or "compiler" in tag.lower() or "source" in tag.lower():
                        audit["project_requirements"][tag] = child.text

            # Check Spring Boot parent version
            parent = root.find("m:parent", ns) or root.find("parent")
            if parent is not None:
                group = parent.find("m:groupId", ns) or parent.find("groupId")
                version = parent.find("m:version", ns) or parent.find("version")
                if group is not None and "spring-boot" in (group.text or ""):
                    audit["project_requirements"]["spring-boot-version"] = version.text if version is not None else "unknown"

        except Exception as e:
            audit["issues"].append(f"Failed to parse pom.xml: {e}")

    # Parse build.gradle for Java version
    for gradle_file in ["build.gradle", "build.gradle.kts"]:
        gradle_path = path / gradle_file
        if gradle_path.exists():
            try:
                content = gradle_path.read_text(encoding="utf-8")
                # Look for sourceCompatibility, targetCompatibility, Java toolchain
                for pattern in [
                    r"sourceCompatibility\s*=\s*['\"]?(\S+?)['\"]?\s",
                    r"targetCompatibility\s*=\s*['\"]?(\S+?)['\"]?\s",
                    r"JavaVersion\.VERSION_(\d+)",
                    r"languageVersion\.set\(JavaLanguageVersion\.of\((\d+)\)\)",
                ]:
                    match = re.search(pattern, content)
                    if match:
                        audit["project_requirements"][f"java_version ({gradle_file})"] = match.group(1)
            except Exception as e:
                audit["issues"].append(f"Failed to parse {gradle_file}: {e}")

    # Version mismatch detection
    if "java" in java_ver.lower():
        java_major = re.search(r'"?(\d+)', java_ver)
        if java_major:
            local_ver = int(java_major.group(1))
            for key, val in audit["project_requirements"].items():
                if val and re.match(r"\d+", str(val)):
                    required = int(re.match(r"\d+", str(val)).group())
                    if required != local_ver and required > 1:
                        audit["issues"].append(
                            f"Java version mismatch: local={local_ver}, project requires {key}={val}"
                        )

    logger.info("_audit_java DONE [%.3fs]", time.perf_counter() - t0)
    return audit


def _audit_python(project_path: str) -> dict:
    """Audit Python environment against project requirements."""
    t0 = time.perf_counter()
    logger.info("_audit_python START for %s", project_path)
    path = Path(project_path)
    audit = {"runtime": {}, "packages": {}, "project_requirements": {}, "issues": []}

    # Check versions concurrently — only check tools that exist on PATH
    cmds = {
        "python": "python --version 2>&1",
        "pip": "pip --version 2>&1",
    }
    # python3 is redundant on Windows (python == python3)
    if not _IS_WINDOWS:
        cmds["python3"] = "python3 --version 2>&1"
    versions = _get_versions(cmds)
    audit["runtime"]["python"] = versions.get("python", "NOT FOUND")
    if "python3" in versions and versions["python3"] != "NOT FOUND":
        audit["runtime"]["python3"] = versions["python3"]
    audit["runtime"]["pip"] = versions.get("pip", "NOT FOUND")
    if _tool_exists("conda"):
        audit["runtime"]["conda"] = _get_version("conda --version 2>&1")

    # Check virtualenv
    venv_active = os.environ.get("VIRTUAL_ENV", None)
    audit["runtime"]["virtualenv_active"] = venv_active or "No virtualenv active"
    if not venv_active:
        audit["issues"].append("No virtualenv active — packages may install globally or conflict")

    # Get installed packages
    result = _run("pip list --format=json 2>&1")
    installed = {}
    if result["returncode"] == 0:
        try:
            packages = json.loads(result["stdout"])
            installed = {p["name"].lower(): p["version"] for p in packages}
        except (json.JSONDecodeError, KeyError):
            pass

    # Parse requirements.txt
    req_path = path / "requirements.txt"
    if req_path.exists():
        try:
            for line in req_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Parse package==version or package>=version
                match = re.match(r"([a-zA-Z0-9_-]+)\s*([><=!~]+)\s*(.+)", line)
                if match:
                    pkg, op, ver = match.groups()
                    pkg_lower = pkg.lower()
                    audit["project_requirements"][pkg] = f"{op}{ver}"
                    if pkg_lower in installed:
                        audit["packages"][pkg] = {"required": f"{op}{ver}", "installed": installed[pkg_lower]}
                    else:
                        audit["packages"][pkg] = {"required": f"{op}{ver}", "installed": "NOT INSTALLED"}
                        audit["issues"].append(f"Package '{pkg}' required but not installed")
                else:
                    pkg = line.split("[")[0].strip().lower()
                    if pkg in installed:
                        audit["packages"][line] = {"required": "any", "installed": installed[pkg]}
                    elif pkg:
                        audit["issues"].append(f"Package '{line}' required but not installed")
        except Exception as e:
            audit["issues"].append(f"Failed to parse requirements.txt: {e}")

    logger.info("_audit_python DONE [%.3fs]", time.perf_counter() - t0)
    return audit


def _audit_dotnet(project_path: str) -> dict:
    """Audit .NET environment against project requirements."""
    t0 = time.perf_counter()
    logger.info("_audit_dotnet START for %s", project_path)
    path = Path(project_path)
    audit = {"runtime": {}, "project_requirements": {}, "issues": []}

    # Check dotnet version, runtimes, SDKs concurrently
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_ver = pool.submit(_get_version, "dotnet --version 2>&1")
        f_runtimes = pool.submit(_run, "dotnet --list-runtimes 2>&1")
        f_sdks = pool.submit(_run, "dotnet --list-sdks 2>&1")

    dotnet_ver = f_ver.result()
    audit["runtime"]["dotnet_sdk"] = dotnet_ver
    rt_result = f_runtimes.result()
    if rt_result["returncode"] == 0:
        audit["runtime"]["runtimes"] = rt_result["stdout"].split("\n")[:10]
    sdk_result = f_sdks.result()
    if sdk_result["returncode"] == 0:
        audit["runtime"]["sdks"] = sdk_result["stdout"].split("\n")[:10]

    # Parse .csproj for target framework
    for csproj in path.glob("*.csproj"):
        try:
            tree = ET.parse(csproj)
            root = tree.getroot()
            tf = root.find(".//TargetFramework")
            if tf is not None and tf.text:
                audit["project_requirements"]["target_framework"] = tf.text
                # Check if local SDK supports it
                if dotnet_ver != "NOT FOUND":
                    sdk_major = re.match(r"(\d+)", dotnet_ver)
                    fw_match = re.search(r"net(\d+)", tf.text)
                    if sdk_major and fw_match:
                        if int(sdk_major.group(1)) < int(fw_match.group(1)):
                            audit["issues"].append(
                                f"SDK version {dotnet_ver} may not support {tf.text}"
                            )

            # Check package references
            for pkg_ref in root.findall(".//PackageReference"):
                name = pkg_ref.get("Include", "")
                version = pkg_ref.get("Version", "")
                if name:
                    audit["project_requirements"][f"nuget:{name}"] = version

        except Exception as e:
            audit["issues"].append(f"Failed to parse {csproj.name}: {e}")

    # Check nuget (only if on PATH)
    if _tool_exists("nuget"):
        audit["runtime"]["nuget"] = _get_version("nuget help 2>&1")

    logger.info("_audit_dotnet DONE [%.3fs]", time.perf_counter() - t0)
    return audit


@mcp.tool()
def audit_local_env(project_path: str) -> dict:
    """Scan the developer's machine for exact dependency versions and compare to project requirements.

    Auto-detects project type (Java/Maven/Gradle, Python/pip, .NET/dotnet) and checks:
    - Runtime versions (Java, Python, dotnet SDK)
    - Build tool versions (Maven, Gradle, pip, dotnet)
    - Environment variables (JAVA_HOME, VIRTUAL_ENV, etc.)
    - Project-declared requirements vs installed versions
    - Version mismatches and missing dependencies

    Args:
        project_path: Absolute path to the project root directory.

    Returns:
        dict: Audit results per detected project type with issues highlighted.
    """
    t0 = time.perf_counter()
    logger.info("=== audit_local_env START for %s ===", project_path)
    if not os.path.isdir(project_path):
        return {"error": f"Directory not found: {project_path}"}

    detected = _detect_project_type(project_path)
    logger.debug("audit_local_env detected types: %s", detected)
    if not detected:
        return {
            "error": "No recognised project type found",
            "hint": "Expected pom.xml, build.gradle, requirements.txt, pyproject.toml, or *.csproj",
        }

    results = {
        "project_path": project_path,
        "detected_types": detected,
        "audits": {},
    }

    seen_types = set()
    for proj in detected:
        ptype = proj["type"]
        if ptype in seen_types:
            continue
        seen_types.add(ptype)

        if ptype == "java":
            results["audits"]["java"] = _audit_java(project_path)
        elif ptype == "python":
            results["audits"]["python"] = _audit_python(project_path)
        elif ptype == "dotnet":
            results["audits"]["dotnet"] = _audit_dotnet(project_path)

    # Collect all issues
    all_issues = []
    for audit in results["audits"].values():
        all_issues.extend(audit.get("issues", []))
    results["total_issues"] = len(all_issues)
    results["all_issues"] = all_issues

    logger.info("=== audit_local_env DONE [%.3fs] ===", time.perf_counter() - t0)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Tool 2: trace_port_conflict
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def trace_port_conflict(port: int) -> dict:
    """Find which process is using a specific port and offer resolution options.

    Useful when a dev server fails to start with "port already in use" errors.

    Args:
        port: The port number to check (e.g. 3000, 8080, 5000).

    Returns:
        dict: Process details (PID, name, command line) using the port, plus resolution suggestions.
    """
    t0 = time.perf_counter()
    logger.info("=== trace_port_conflict START port=%d ===", port)
    conflicts = []

    # Windows: netstat + tasklist
    result = _run(f'netstat -ano | findstr ":{port} "')
    if result["returncode"] == 0 and result["stdout"]:
        seen_pids = set()
        for line in result["stdout"].split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                local_addr = parts[1]
                state = parts[3] if len(parts) > 3 else ""
                pid = parts[-1]

                if pid in seen_pids or not pid.isdigit():
                    continue
                seen_pids.add(pid)

                # Get process name
                proc_result = _run(f'tasklist /FI "PID eq {pid}" /FO CSV /NH')
                proc_name = "unknown"
                cmd_line = ""
                if proc_result["returncode"] == 0 and proc_result["stdout"]:
                    csv_parts = proc_result["stdout"].split(",")
                    if csv_parts:
                        proc_name = csv_parts[0].strip('"')

                # Try to get command line via WMIC
                wmic_result = _run(f'wmic process where ProcessId={pid} get CommandLine /FORMAT:LIST')
                if wmic_result["returncode"] == 0:
                    for wline in wmic_result["stdout"].split("\n"):
                        if wline.startswith("CommandLine="):
                            cmd_line = wline.split("=", 1)[1].strip()

                conflicts.append({
                    "pid": int(pid),
                    "process_name": proc_name,
                    "command_line": cmd_line[:200] if cmd_line else "",
                    "local_address": local_addr,
                    "state": state,
                })

    logger.info("=== trace_port_conflict DONE [%.3fs] ===", time.perf_counter() - t0)
    if not conflicts:
        return {"port": port, "status": "free", "message": f"Port {port} is not in use"}

    return {
        "port": port,
        "status": "in_use",
        "conflicts": conflicts,
        "resolution_options": [
            f"Kill the process: taskkill /PID {conflicts[0]['pid']} /F",
            f"Use a different port in your config",
            f"Check if this is a legitimate service that should be running",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool 3: docker_container_logs
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def docker_container_logs(
    container: str = "",
    tail: int = 100,
    since: str = "",
    list_containers: bool = False,
) -> dict:
    """Get logs from Docker containers to debug silent backend crashes.

    Args:
        container: Container name or ID. Leave empty with list_containers=True to see running containers.
        tail: Number of lines from end of logs (default: 100).
        since: Only logs since this time (e.g. '5m', '1h', '2024-01-01T00:00:00').
        list_containers: If True, list all running containers instead of fetching logs.

    Returns:
        dict: Container logs or list of running containers.
    """
    t0 = time.perf_counter()
    logger.info("=== docker_container_logs START container=%s ===", container or "(list)")
    # Check Docker is available
    docker_check = _run("docker --version")
    if docker_check["returncode"] != 0:
        return {"error": "Docker is not installed or not in PATH"}

    if list_containers:
        result = _run('docker ps --format "{{.ID}}\\t{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"')
        if result["returncode"] != 0:
            return {"error": f"Failed to list containers: {result['stderr']}"}

        containers = []
        for line in result["stdout"].split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                containers.append({
                    "id": parts[0],
                    "name": parts[1],
                    "image": parts[2],
                    "status": parts[3],
                    "ports": parts[4] if len(parts) > 4 else "",
                })

        return {"running_containers": containers, "count": len(containers)}

    if not container:
        return {"error": "Provide a container name/ID, or set list_containers=True"}

    cmd = f"docker logs --tail {tail}"
    if since:
        cmd += f" --since {since}"
    cmd += f" {container}"

    result = _run(cmd, timeout=15)
    # Docker logs go to stderr for some containers
    logs = result["stdout"] or result["stderr"]

    if result["returncode"] != 0 and "No such container" in logs:
        return {"error": f"Container '{container}' not found. Use list_containers=True to see running containers."}

    # Extract error patterns
    error_lines = [
        line for line in logs.split("\n")
        if any(kw in line.lower() for kw in ["error", "exception", "fatal", "failed", "panic", "traceback"])
    ]

    logger.info("=== docker_container_logs DONE [%.3fs] ===", time.perf_counter() - t0)
    return {
        "container": container,
        "log_lines": len(logs.split("\n")),
        "logs": logs,
        "errors_detected": len(error_lines),
        "error_lines": error_lines[:20] if error_lines else [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool 4: diagnose_project
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def diagnose_project(project_path: str) -> dict:
    """Run a full health check on a project — build, common issues, and environment.

    Checks for:
    - Missing dependency directories (node_modules, .venv, target, bin/obj)
    - Lock file consistency
    - Common misconfigurations
    - Git status and branch info
    - Available build/run commands

    Args:
        project_path: Absolute path to the project root directory.

    Returns:
        dict: Health check results with issues and recommendations.
    """
    t0 = time.perf_counter()
    logger.info("=== diagnose_project START for %s ===", project_path)
    if not os.path.isdir(project_path):
        return {"error": f"Directory not found: {project_path}"}

    path = Path(project_path)
    detected = _detect_project_type(project_path)
    health = {
        "project_path": project_path,
        "detected_types": detected,
        "checks": [],
        "issues": [],
        "recommendations": [],
    }

    # Git status (run in project directory)
    git_result = _run("git status --porcelain", cwd=project_path)
    git_branch = _run("git branch --show-current", cwd=project_path)
    if git_branch["returncode"] == 0:
        health["git"] = {
            "branch": git_branch["stdout"],
            "dirty_files": len([l for l in git_result["stdout"].split("\n") if l.strip()]) if git_result["stdout"] else 0,
        }

    for proj in detected:
        ptype = proj["type"]

        if ptype == "java":
            # Check target/build directory
            if not (path / "target").exists() and not (path / "build").exists():
                health["issues"].append("No target/ or build/ directory — project may not have been compiled")
                health["recommendations"].append("Run: mvn clean install OR gradle build")

            # Check .mvn/wrapper
            if (path / "mvnw").exists() or (path / "mvnw.cmd").exists():
                health["checks"].append("Maven wrapper found — use ./mvnw instead of mvn")

            # Check gradle wrapper
            if (path / "gradlew").exists() or (path / "gradlew.bat").exists():
                health["checks"].append("Gradle wrapper found — use ./gradlew instead of gradle")

        elif ptype == "python":
            # Check virtualenv
            has_venv = (path / ".venv").exists() or (path / "venv").exists()
            if not has_venv and not os.environ.get("VIRTUAL_ENV"):
                health["issues"].append("No virtualenv found and none active — risk of global package conflicts")
                health["recommendations"].append("Run: python -m venv .venv && .venv\\Scripts\\activate")

            # Check __pycache__ pollution (cap iteration to avoid slow scans)
            pycache_count = 0
            for _ in path.rglob("__pycache__"):
                pycache_count += 1
                if pycache_count > 20:
                    break
            if pycache_count > 20:
                health["checks"].append(f"Found {pycache_count}+ __pycache__ directories — consider cleanup")

        elif ptype == "dotnet":
            # Check bin/obj
            if not (path / "bin").exists() and not (path / "obj").exists():
                health["issues"].append("No bin/ or obj/ directory — project may not have been built")
                health["recommendations"].append("Run: dotnet build")

            # Check for global.json
            if (path / "global.json").exists():
                try:
                    gj = json.loads((path / "global.json").read_text(encoding="utf-8"))
                    sdk_ver = gj.get("sdk", {}).get("version", "")
                    if sdk_ver:
                        health["checks"].append(f"global.json pins SDK to {sdk_ver}")
                except Exception:
                    pass

    # Check for Docker
    if (path / "Dockerfile").exists() or (path / "docker-compose.yml").exists() or (path / "docker-compose.yaml").exists():
        health["checks"].append("Docker configuration found")
        if not _tool_exists("docker"):
            health["issues"].append("Dockerfile found but Docker is not installed or not in PATH")

    # Check for .env file
    if (path / ".env").exists():
        health["checks"].append(".env file found")
    if (path / ".env.example").exists() and not (path / ".env").exists():
        health["issues"].append(".env.example exists but no .env file — copy and configure it")
        health["recommendations"].append("Run: copy .env.example .env")

    health["total_issues"] = len(health["issues"])
    logger.info("=== diagnose_project DONE [%.3fs] ===", time.perf_counter() - t0)
    return health


# ═══════════════════════════════════════════════════════════════════════════
# Tool 5: check_env_vars
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def check_env_vars(names: str) -> dict:
    """Check if specific environment variables are set and show their values.

    Useful for debugging configuration issues (masked for sensitive values).

    Args:
        names: Comma-separated environment variable names to check (e.g. 'JAVA_HOME,PATH,ASPNETCORE_ENVIRONMENT').

    Returns:
        dict: Status of each environment variable.
    """
    t0 = time.perf_counter()
    logger.info("=== check_env_vars START names=%s ===", names)
    var_names = [n.strip() for n in names.split(",") if n.strip()]
    results = {}
    sensitive_keywords = ["password", "secret", "token", "key", "credential", "api_key"]

    for name in var_names:
        value = os.environ.get(name)
        if value is None:
            results[name] = {"status": "NOT SET", "value": None}
        elif any(kw in name.lower() for kw in sensitive_keywords):
            results[name] = {"status": "SET", "value": f"***({len(value)} chars)"}
        else:
            # Truncate very long values like PATH
            display_value = value if len(value) <= 200 else value[:200] + "..."
            results[name] = {"status": "SET", "value": display_value}

    missing = [n for n, v in results.items() if v["status"] == "NOT SET"]
    logger.info("=== check_env_vars DONE [%.3fs] ===", time.perf_counter() - t0)
    return {
        "variables": results,
        "total_checked": len(var_names),
        "missing": missing,
        "missing_count": len(missing),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tool 6: analyze_stacktrace
# ═══════════════════════════════════════════════════════════════════════════

# Common error patterns → (root_cause, fix_suggestion)
_ERROR_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # ── Java ──
    (re.compile(r"java\.lang\.OutOfMemoryError", re.I),
     "JVM ran out of heap memory",
     "Increase heap size: -Xmx512m or -Xmx1g. Check for memory leaks (unclosed streams, growing collections)."),
    (re.compile(r"java\.lang\.ClassNotFoundException:\s*(.+)", re.I),
     "Class not found on classpath",
     "Ensure the dependency containing '{match}' is declared in pom.xml/build.gradle and the JAR is on the classpath."),
    (re.compile(r"java\.lang\.NoClassDefFoundError:\s*(.+)", re.I),
     "Class was available at compile time but missing at runtime",
     "Check for dependency version conflicts. Run 'mvn dependency:tree' to inspect. Ensure transitive deps are not excluded."),
    (re.compile(r"java\.lang\.NullPointerException", re.I),
     "Null reference accessed",
     "Check the line indicated in the stack trace. Add null checks or use Optional. Review object initialization order."),
    (re.compile(r"java\.lang\.StackOverflowError", re.I),
     "Infinite recursion detected",
     "Check recursive method calls in the stack trace. Add a proper base case or convert to iteration."),
    (re.compile(r"java\.net\.BindException.*Address already in use", re.I),
     "Port already in use",
     "Another process is using the port. Use the trace_port_conflict tool or change the port in application config."),
    (re.compile(r"java\.sql\.SQLException.*Communications link failure", re.I),
     "Database connection failed",
     "Check database is running, hostname/port are correct, and network/firewall allows the connection."),
    (re.compile(r"BeanCreationException.*(?:NoSuchBeanDefinitionException|UnsatisfiedDependencyException)", re.I),
     "Spring bean wiring failure",
     "Check @Component/@Service/@Repository annotations, component scan packages, and @Autowired dependencies."),
    (re.compile(r"javax\.net\.ssl\.SSLHandshakeException", re.I),
     "SSL/TLS handshake failure",
     "Check certificate validity, truststore config (-Djavax.net.ssl.trustStore), or add the cert to the JVM cacerts."),

    # ── Python ──
    (re.compile(r"ModuleNotFoundError:\s*No module named '(.+)'", re.I),
     "Python module not installed",
     "Run 'pip install {match}'. If using a virtualenv, ensure it is activated."),
    (re.compile(r"ImportError:\s*cannot import name '(.+)' from '(.+)'", re.I),
     "Import name not found in module",
     "Check for version mismatch — the installed version of '{match2}' may not have '{match}'. Run 'pip show {match2}'."),
    (re.compile(r"FileNotFoundError.*No such file or directory:\s*'?(.+?)'?$", re.I | re.M),
     "File or directory not found",
     "Verify the path '{match}' exists. Check working directory (os.getcwd()) and relative vs absolute paths."),
    (re.compile(r"PermissionError.*Permission denied:\s*'?(.+?)'?$", re.I | re.M),
     "Insufficient file permissions",
     "Check file/folder permissions on '{match}'. On Windows, run as Administrator if needed."),
    (re.compile(r"ConnectionRefusedError", re.I),
     "Connection refused by target host",
     "Ensure the target service is running and the host:port is correct."),
    (re.compile(r"KeyError:\s*'?(.+?)'?$", re.I | re.M),
     "Dictionary key not found",
     "Key '{match}' does not exist. Use .get(key, default) or check key existence before access."),
    (re.compile(r"TypeError:\s*(.+)", re.I),
     "Type mismatch in operation",
     "Check argument types at the indicated line. Common cause: passing None where a value is expected."),

    # ── .NET ──
    (re.compile(r"System\.NullReferenceException", re.I),
     "Null reference accessed",
     "Check the variable at the indicated line. Use null-conditional (?.) or add null checks."),
    (re.compile(r"System\.IO\.FileNotFoundException.*'(.+)'", re.I),
     "File not found",
     "Verify the file path '{match}' exists and the application has read access."),
    (re.compile(r"System\.InvalidOperationException.*Sequence contains no elements", re.I),
     "LINQ query returned no results but .First()/.Single() was used",
     "Use .FirstOrDefault() or .SingleOrDefault() and check for null."),
    (re.compile(r"System\.Data\.SqlClient\.SqlException.*Cannot open database", re.I),
     "SQL Server connection failed",
     "Check the connection string in appsettings.json, database name, and server availability."),

    # ── Node.js ──
    (re.compile(r"Error: Cannot find module '(.+)'", re.I),
     "Node module not found",
     "Run 'npm install' or 'npm install {match}'. Check package.json for the dependency."),
    (re.compile(r"EADDRINUSE.*:(\d+)", re.I),
     "Port already in use",
     "Port {match} is occupied. Kill the existing process or use a different port."),
    (re.compile(r"ECONNREFUSED", re.I),
     "Connection refused by target service",
     "Ensure the backend/database service is running and the URL/port is correct."),

    # ── Generic ──
    (re.compile(r"Out [Oo]f [Mm]emory", re.I),
     "Process ran out of memory",
     "Increase available memory or investigate memory leaks. Check for large data structures held in memory."),
    (re.compile(r"Permission denied|Access is denied", re.I),
     "Insufficient permissions",
     "Run with elevated privileges or fix file/folder permissions."),
    (re.compile(r"Connection refused|ECONNREFUSED", re.I),
     "Target service not reachable",
     "Ensure the service is running, check host/port, and verify firewall rules."),
    (re.compile(r"Timeout|timed? ?out", re.I),
     "Operation timed out",
     "Increase timeout settings or investigate why the target is slow (network, load, deadlock)."),
]


def _detect_language(error_text: str) -> str:
    """Detect the programming language from error/stacktrace patterns."""
    if re.search(r"\bat\s+[\w$.]+\([\w]+\.java:\d+\)", error_text):
        return "java"
    if re.search(r"Traceback \(most recent call last\)|File \".*\.py\"", error_text):
        return "python"
    if re.search(r"\bat\s+[\w.]+\s+in\s+.+\.cs:line\s+\d+", error_text) or "System." in error_text:
        return "dotnet"
    if re.search(r"\bat\s+[\w.]+\s+\(.+\.js:\d+:\d+\)", error_text) or re.search(r"\bat\s+[\w.]+\s+\(.+\.ts:\d+:\d+\)", error_text):
        return "nodejs"
    return "unknown"


def _extract_exception_info(error_text: str, language: str) -> dict:
    """Extract exception class, message, and key stack frames."""
    info: dict = {"exception_type": None, "message": None, "key_frames": []}

    if language == "java":
        # Match: com.example.SomeException: message
        m = re.search(r"^([\w$.]+(?:Exception|Error|Throwable)):\s*(.+)$", error_text, re.M)
        if not m:
            m = re.search(r"Caused by:\s*([\w$.]+(?:Exception|Error|Throwable)):\s*(.+)$", error_text, re.M)
        if m:
            info["exception_type"] = m.group(1)
            info["message"] = m.group(2).strip()
        # Extract frames
        for fm in re.finditer(r"\bat\s+([\w$.]+)\(([\w]+\.java):(\d+)\)", error_text):
            info["key_frames"].append({"method": fm.group(1), "file": fm.group(2), "line": int(fm.group(3))})
            if len(info["key_frames"]) >= 10:
                break
        # Find "Caused by" chain
        caused_by = re.findall(r"Caused by:\s*([\w$.]+(?:Exception|Error)):\s*(.+)", error_text)
        if caused_by:
            info["root_exception"] = caused_by[-1][0]
            info["root_message"] = caused_by[-1][1].strip()

    elif language == "python":
        # Last exception line is usually the actual error
        m = re.search(r"^(\w+(?:Error|Exception|Warning)):\s*(.+)$", error_text, re.M)
        if m:
            info["exception_type"] = m.group(1)
            info["message"] = m.group(2).strip()
        # Extract frames: File "path", line N, in func
        for fm in re.finditer(r'File "(.+?)", line (\d+), in (.+)', error_text):
            info["key_frames"].append({"file": fm.group(1), "line": int(fm.group(2)), "method": fm.group(3)})
            if len(info["key_frames"]) >= 10:
                break

    elif language == "dotnet":
        m = re.search(r"(System\.[\w.]+(?:Exception|Error)):\s*(.+?)(?:\r?\n|$)", error_text)
        if not m:
            m = re.search(r"([\w.]+Exception):\s*(.+?)(?:\r?\n|$)", error_text)
        if m:
            info["exception_type"] = m.group(1)
            info["message"] = m.group(2).strip()
        for fm in re.finditer(r"at\s+([\w.<>]+)\s+in\s+(.+?):line\s+(\d+)", error_text):
            info["key_frames"].append({"method": fm.group(1), "file": fm.group(2), "line": int(fm.group(3))})
            if len(info["key_frames"]) >= 10:
                break

    elif language == "nodejs":
        m = re.search(r"^(\w*Error):\s*(.+)$", error_text, re.M)
        if m:
            info["exception_type"] = m.group(1)
            info["message"] = m.group(2).strip()
        for fm in re.finditer(r"at\s+(?:(\S+)\s+)?\((.+?):(\d+):\d+\)", error_text):
            info["key_frames"].append({"method": fm.group(1) or "<anonymous>", "file": fm.group(2), "line": int(fm.group(3))})
            if len(info["key_frames"]) >= 10:
                break

    return info


def _match_error_patterns(error_text: str) -> list[dict]:
    """Match error text against known patterns and return diagnoses."""
    matches = []
    for pattern, root_cause, fix in _ERROR_PATTERNS:
        m = pattern.search(error_text)
        if m:
            # Substitute captured groups into the strings
            rc = root_cause
            fx = fix
            if m.lastindex and m.lastindex >= 1:
                rc = rc.replace("{match}", m.group(1))
                fx = fx.replace("{match}", m.group(1))
            if m.lastindex and m.lastindex >= 2:
                rc = rc.replace("{match2}", m.group(2))
                fx = fx.replace("{match2}", m.group(2))
            matches.append({"pattern": pattern.pattern[:80], "root_cause": rc, "suggested_fix": fx})
    return matches


@mcp.tool()
def analyze_stacktrace(
    error_text: str = "",
    log_file: str = "",
    tail_lines: int = 100,
) -> dict:
    """Read an error or stack trace and find the root cause with suggested fix.

    Accepts error text directly (e.g. copied from a run/terminal window) or
    reads the tail of a log file. Supports Java, Python, .NET, and Node.js
    stack traces plus common error patterns (OOM, port conflicts, permissions, etc.).

    Args:
        error_text: The error output or stack trace text to analyze.
        log_file: Path to a log file to read errors from (alternative to error_text).
        tail_lines: Number of lines to read from the end of log_file (default: 100).

    Returns:
        dict: Analysis with detected language, exception info, root cause, and suggested fixes.
    """
    t0 = time.perf_counter()
    logger.info("=== analyze_stacktrace START ===")

    # Resolve input
    if not error_text and not log_file:
        return {"error": "Provide either error_text or log_file path."}

    if log_file:
        log_path = Path(log_file)
        if not log_path.is_file():
            return {"error": f"Log file not found: {log_file}"}
        try:
            all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            error_text = "\n".join(all_lines[-tail_lines:])
            logger.debug("Read %d lines (last %d) from %s", len(all_lines), tail_lines, log_file)
        except Exception as e:
            return {"error": f"Failed to read log file: {e}"}

    # Detect language
    language = _detect_language(error_text)

    # Extract exception details
    exception_info = _extract_exception_info(error_text, language)

    # Match against known patterns
    pattern_matches = _match_error_patterns(error_text)

    # Build result
    result: dict = {
        "detected_language": language,
        "exception_type": exception_info.get("exception_type"),
        "error_message": exception_info.get("message"),
        "key_stack_frames": exception_info.get("key_frames", []),
    }

    if exception_info.get("root_exception"):
        result["root_exception"] = exception_info["root_exception"]
        result["root_message"] = exception_info["root_message"]

    if pattern_matches:
        result["diagnoses"] = pattern_matches
        result["primary_root_cause"] = pattern_matches[0]["root_cause"]
        result["primary_fix"] = pattern_matches[0]["suggested_fix"]
    else:
        result["diagnoses"] = []
        result["primary_root_cause"] = "No known pattern matched — review the exception type and message above."
        result["primary_fix"] = "Search for the exception type and message online, or check the source at the top stack frame."

    if log_file:
        result["source_file"] = log_file
        result["lines_analyzed"] = tail_lines

    logger.info("=== analyze_stacktrace DONE [%.3fs] lang=%s exc=%s matches=%d ===",
                time.perf_counter() - t0, language,
                exception_info.get("exception_type", "?"), len(pattern_matches))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Tool 7: run_and_analyze
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def run_and_analyze(
    command: str,
    cwd: str = "",
    timeout: int = 120,
    tail_lines: int = 200,
) -> dict:
    """Run a build/start command, capture its output, and auto-analyze any errors.

    Executes the command in a subprocess, captures stdout and stderr, then runs
    stack trace analysis on the output if the command fails. Useful for running
    builds (mvn, gradle, dotnet, pip, npm) and immediately diagnosing failures.

    Args:
        command: The command to run (e.g. 'mvn clean install', 'python app.py', 'dotnet build').
        cwd: Working directory for the command. Defaults to current directory.
        timeout: Max seconds to wait for the command to finish (default: 120).
        tail_lines: Number of output lines to include in the analysis (default: 200).

    Returns:
        dict: Command result with exit code, output, and error analysis if the command failed.
    """
    t0 = time.perf_counter()
    logger.info("=== run_and_analyze START cmd=%s cwd=%s ===", command[:120], cwd or "(default)")

    if not command.strip():
        return {"error": "No command provided."}

    work_dir = cwd if cwd and os.path.isdir(cwd) else None

    # Run the command with extended timeout
    result = _run(command, shell=True, timeout=timeout, cwd=work_dir)
    elapsed = time.perf_counter() - t0

    # Combine stdout and stderr for full output
    full_output = ""
    if result["stdout"]:
        full_output += result["stdout"]
    if result["stderr"]:
        if full_output:
            full_output += "\n"
        full_output += result["stderr"]

    # Trim to tail_lines
    output_lines = full_output.splitlines()
    total_lines = len(output_lines)
    if total_lines > tail_lines:
        trimmed_output = "\n".join(output_lines[-tail_lines:])
        trimmed = True
    else:
        trimmed_output = full_output
        trimmed = False

    response: dict = {
        "command": command,
        "cwd": work_dir or os.getcwd(),
        "exit_code": result["returncode"],
        "success": result["returncode"] == 0,
        "elapsed_seconds": round(elapsed, 2),
        "total_output_lines": total_lines,
        "output": trimmed_output,
    }

    if trimmed:
        response["note"] = f"Output trimmed to last {tail_lines} of {total_lines} lines."

    # If the command failed, auto-analyze the output
    if result["returncode"] != 0:
        language = _detect_language(full_output)
        exception_info = _extract_exception_info(full_output, language)
        pattern_matches = _match_error_patterns(full_output)

        analysis: dict = {
            "detected_language": language,
            "exception_type": exception_info.get("exception_type"),
            "error_message": exception_info.get("message"),
            "key_stack_frames": exception_info.get("key_frames", []),
        }

        if exception_info.get("root_exception"):
            analysis["root_exception"] = exception_info["root_exception"]
            analysis["root_message"] = exception_info["root_message"]

        if pattern_matches:
            analysis["diagnoses"] = pattern_matches
            analysis["primary_root_cause"] = pattern_matches[0]["root_cause"]
            analysis["primary_fix"] = pattern_matches[0]["suggested_fix"]
        else:
            # Extract error-like lines as fallback
            error_lines = [
                line.strip() for line in output_lines
                if any(kw in line.lower() for kw in ["error", "exception", "fatal", "failed", "failure"])
                and len(line.strip()) > 10
            ]
            analysis["error_lines"] = error_lines[:20] if error_lines else []
            analysis["primary_root_cause"] = "No known pattern matched — review error lines above."
            analysis["primary_fix"] = "Check the error lines and the full output for details."

        response["analysis"] = analysis

    logger.info("=== run_and_analyze DONE [%.3fs] rc=%d ===", elapsed, result["returncode"])
    return response


if __name__ == "__main__":
    mcp.run()
