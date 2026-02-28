# Doctor Context MCP Server

> Gives your AI coding agent **"eyes" and "hands"** inside your local development environment.  
> Solves the **"Why won't this run on my machine?"** problem for **Java**, **Python**, and **.NET** projects.

Most AI coding agents write "correct" code that won't run because they don't understand your local environment — OS versions, conflicting dependencies, broken environment variables, or Docker issues. Doctor Context bridges that gap by turning the AI from a "code writer" into a **Senior DevOps Partner** that handles the plumbing.

---

## Features

- 🔍 **Environment Auditing** — Scans your machine for exact runtime/build tool versions and compares them against project requirements
- 🔌 **Port Conflict Tracing** — Finds exactly which process is blocking a port and offers resolution
- 🐳 **Docker Log Access** — Fetches container logs with automatic error/exception detection
- 🏥 **Project Health Checks** — Diagnoses missing build artifacts, virtualenvs, `.env` files, and common misconfigurations
- 🔐 **Environment Variable Checks** — Validates required env vars are set, with automatic masking of sensitive values
- 🔥 **Stack Trace Analysis** — Parses errors/stack traces from run windows or log files, identifies root cause, and suggests fixes (Java, Python, .NET, Node.js)

---

## Supported Project Types

| Type | Detected By | What's Checked |
|------|-------------|----------------|
| **Java** | `pom.xml`, `build.gradle`, `build.gradle.kts` | Java/javac version, JAVA_HOME, Maven/Gradle versions, `maven.compiler.source/target`, Spring Boot parent version, Java toolchain config |
| **Python** | `requirements.txt`, `pyproject.toml`, `Pipfile` | Python version, pip, virtualenv status, conda, installed packages vs requirements, missing packages |
| **.NET** | `*.csproj`, `*.sln` | dotnet SDK version, installed runtimes/SDKs, target framework from `.csproj`, NuGet packages, `global.json` SDK pinning |

---

## Tools

### 1. `audit_local_env`

Scans the developer's machine for exact dependency versions and compares them to the project's declared requirements.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_path` | string | ✅ | Absolute path to the project root directory |

**What it checks:**

- **Java**: `java -version`, `javac`, `JAVA_HOME`, `mvn`, `gradle`, parses `pom.xml` for compiler source/target and Spring Boot version, parses `build.gradle` for `sourceCompatibility`/`targetCompatibility`/toolchain, detects version mismatches
- **Python**: `python --version`, `pip`, virtualenv status, `conda`, compares `pip list` against `requirements.txt`, flags missing packages
- **.NET**: `dotnet --version`, installed SDKs and runtimes, parses `.csproj` for `TargetFramework` and `PackageReference` entries, detects SDK/framework incompatibilities

**Example output:**

```json
{
  "project_path": "C:\\Projects\\my-api",
  "detected_types": [{"type": "java", "build_tool": "maven", "marker": "pom.xml"}],
  "audits": {
    "java": {
      "runtime": {
        "java": "openjdk version \"21.0.1\"",
        "javac": "javac 21.0.1",
        "JAVA_HOME": "C:\\Program Files\\Java\\jdk-21"
      },
      "build_tool": {
        "maven": "Apache Maven 3.9.6",
        "gradle": "NOT FOUND"
      },
      "project_requirements": {
        "maven.compiler.source": "11",
        "maven.compiler.target": "11",
        "spring-boot-version": "2.7.18"
      },
      "issues": [
        "Java version mismatch: local=21, project requires maven.compiler.source=11"
      ]
    }
  },
  "total_issues": 1,
  "all_issues": ["Java version mismatch: local=21, project requires maven.compiler.source=11"]
}
```

---

### 2. `trace_port_conflict`

Finds which process is using a specific port and provides resolution options. Essential when a dev server fails to start with "port already in use" errors.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `port` | integer | ✅ | The port number to check (e.g. 3000, 8080, 5000) |

**What it does:**

- Runs `netstat -ano` to find processes bound to the port
- Looks up process name via `tasklist`
- Fetches the full command line via `wmic`
- Provides ready-to-run kill commands

**Example output:**

```json
{
  "port": 8080,
  "status": "in_use",
  "conflicts": [
    {
      "pid": 12456,
      "process_name": "java.exe",
      "command_line": "java -jar target/my-api-1.0.0.jar --server.port=8080",
      "local_address": "0.0.0.0:8080",
      "state": "LISTENING"
    }
  ],
  "resolution_options": [
    "Kill the process: taskkill /PID 12456 /F",
    "Use a different port in your config",
    "Check if this is a legitimate service that should be running"
  ]
}
```

---

### 3. `docker_container_logs`

Fetches logs from Docker containers with automatic error/exception detection. Replaces the need to manually run `docker logs` and scan for issues.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `container` | string | ❌ | `""` | Container name or ID |
| `tail` | integer | ❌ | `100` | Number of lines from end of logs |
| `since` | string | ❌ | `""` | Time filter (e.g. `5m`, `1h`, `2024-01-01T00:00:00`) |
| `list_containers` | boolean | ❌ | `false` | List all running containers instead |

**Features:**

- Auto-detects error patterns: `error`, `exception`, `fatal`, `failed`, `panic`, `traceback`
- Extracts and highlights error lines separately
- Handles containers that log to stderr

**Example — List running containers:**

```json
// Input: list_containers=true
{
  "running_containers": [
    {"id": "a1b2c3", "name": "my-api", "image": "openjdk:21", "status": "Up 2 hours", "ports": "0.0.0.0:8080->8080/tcp"}
  ],
  "count": 1
}
```

**Example — Fetch logs with error detection:**

```json
// Input: container="my-api", tail=50
{
  "container": "my-api",
  "log_lines": 50,
  "logs": "...",
  "errors_detected": 2,
  "error_lines": [
    "ERROR 2026-02-28 10:15:32 - Failed to connect to database",
    "java.sql.SQLException: Connection refused"
  ]
}
```

---

### 4. `diagnose_project`

Runs a comprehensive health check on a project directory — build state, common misconfigurations, and environment.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_path` | string | ✅ | Absolute path to the project root directory |

**What it checks:**

| Check | Java | Python | .NET | All |
|-------|------|--------|------|-----|
| Build output exists | `target/` or `build/` | — | `bin/` or `obj/` | — |
| Build wrapper present | `mvnw` / `gradlew` | — | — | — |
| Virtualenv active | — | `.venv/` or `VIRTUAL_ENV` | — | — |
| `__pycache__` pollution | — | Count > 20 | — | — |
| SDK pinning | — | — | `global.json` | — |
| `.env` file | — | — | — | `.env` missing but `.env.example` exists |
| Docker config | — | — | — | `Dockerfile` / `docker-compose.yml` |
| Git status | — | — | — | Branch name, dirty file count |

**Example output:**

```json
{
  "project_path": "C:\\Projects\\my-api",
  "detected_types": [{"type": "java", "build_tool": "maven", "marker": "pom.xml"}],
  "git": {"branch": "feature/PTRT-4447", "dirty_files": 3},
  "checks": [
    "Maven wrapper found — use ./mvnw instead of mvn",
    "Docker configuration found"
  ],
  "issues": [
    ".env.example exists but no .env file — copy and configure it"
  ],
  "recommendations": [
    "Run: copy .env.example .env"
  ],
  "total_issues": 1
}
```

---

### 5. `check_env_vars`

Checks whether specific environment variables are set and displays their values. Automatically masks sensitive values (passwords, tokens, keys, secrets).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `names` | string | ✅ | Comma-separated env var names (e.g. `JAVA_HOME,PATH,DB_PASSWORD`) |

**Features:**

- Auto-detects sensitive variables by name (`password`, `secret`, `token`, `key`, `credential`, `api_key`)
- Masks sensitive values showing only character count
- Truncates long values (e.g. `PATH`) to 200 characters

**Example output:**

```json
{
  "variables": {
    "JAVA_HOME": {"status": "SET", "value": "C:\\Program Files\\Java\\jdk-21"},
    "DB_PASSWORD": {"status": "SET", "value": "***(24 chars)"},
    "MISSING_VAR": {"status": "NOT SET", "value": null}
  },
  "total_checked": 3,
  "missing": ["MISSING_VAR"],
  "missing_count": 1
}
```

### 6. `analyze_stacktrace`

Reads an error or stack trace and finds the root cause with a suggested fix. Accepts error text directly (e.g. copied from a run/terminal window) or reads the tail of a log file.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `error_text` | string | ❌ | `""` | The error output or stack trace text to analyze |
| `log_file` | string | ❌ | `""` | Path to a log file to read errors from (alternative to `error_text`) |
| `tail_lines` | integer | ❌ | `100` | Number of lines to read from the end of `log_file` |

> **Note:** Provide either `error_text` or `log_file` — at least one is required.

**Supported languages & patterns:**

| Language | Patterns Detected |
|----------|-------------------|
| **Java** | `OutOfMemoryError`, `ClassNotFoundException`, `NoClassDefFoundError`, `NullPointerException`, `StackOverflowError`, `BindException`, `SQLException`, `BeanCreationException`, `SSLHandshakeException` |
| **Python** | `ModuleNotFoundError`, `ImportError`, `FileNotFoundError`, `PermissionError`, `ConnectionRefusedError`, `KeyError`, `TypeError` |
| **.NET** | `NullReferenceException`, `FileNotFoundException`, `InvalidOperationException`, `SqlException` |
| **Node.js** | `Cannot find module`, `EADDRINUSE`, `ECONNREFUSED` |
| **Generic** | Out of memory, permission denied, connection refused, timeouts |

**Example — Analyze error text:**

```json
{
  "detected_language": "java",
  "exception_type": "java.lang.ClassNotFoundException",
  "error_message": "com.example.MyService",
  "key_stack_frames": [
    {"method": "java.lang.ClassLoader.loadClass", "file": "ClassLoader.java", "line": 521}
  ],
  "diagnoses": [
    {
      "root_cause": "Class not found on classpath",
      "suggested_fix": "Ensure the dependency containing 'com.example.MyService' is declared in pom.xml/build.gradle and the JAR is on the classpath."
    }
  ],
  "primary_root_cause": "Class not found on classpath",
  "primary_fix": "Ensure the dependency containing 'com.example.MyService' is declared in pom.xml/build.gradle and the JAR is on the classpath."
}
```

**Example — Analyze from log file:**

```json
// Input: log_file="C:\\Projects\\my-api\\logs\\app.log", tail_lines=200
{
  "detected_language": "python",
  "exception_type": "ModuleNotFoundError",
  "error_message": "No module named 'requests'",
  "key_stack_frames": [
    {"file": "app.py", "line": 3, "method": "<module>"}
  ],
  "diagnoses": [
    {
      "root_cause": "Python module not installed",
      "suggested_fix": "Run 'pip install requests'. If using a virtualenv, ensure it is activated."
    }
  ],
  "primary_root_cause": "Python module not installed",
  "primary_fix": "Run 'pip install requests'. If using a virtualenv, ensure it is activated.",
  "source_file": "C:\\Projects\\my-api\\logs\\app.log",
  "lines_analyzed": 200
}
```

---

## Installation

### Prerequisites

- Python 3.11+
- `mcp` package (`pip install mcp`)

### Setup

```bash
cd c:\Workspace\OtherRepos\mcp-servers\doctor-context-mcp
pip install -e .
```

### Register with Amp

Add to your Amp MCP settings file (`~/.config/amp/settings.json`):

```json
{
  "mcpServers": {
    "doctor-context": {
      "command": "python",
      "args": ["-m", "src"],
      "cwd": "c:\\Workspace\\OtherRepos\\mcp-servers\\doctor-context-mcp"
    }
  }
}
```

Restart Amp to activate.

### Register with VS Code (Copilot / MCP extension)

Add to your VS Code `settings.json` (`Ctrl+Shift+P` → "Preferences: Open User Settings (JSON)"):

```json
{
  "mcp": {
    "servers": {
      "doctor-context": {
        "command": "python",
        "args": ["-m", "src"],
        "cwd": "c:\\Workspace\\OtherRepos\\mcp-servers\\doctor-context-mcp"
      }
    }
  }
}
```

Or add a `.vscode/mcp.json` file in your project root for project-scoped registration:

```json
{
  "servers": {
    "doctor-context": {
      "command": "python",
      "args": ["-m", "src"],
      "cwd": "c:\\Workspace\\OtherRepos\\mcp-servers\\doctor-context-mcp"
    }
  }
}
```

Reload VS Code to activate.

---

## Usage Examples

### "My Spring Boot app won't start — wrong Java version?"

```
> Audit my project at C:\Projects\my-spring-api
```

The AI calls `audit_local_env` → discovers Java 21 installed but `pom.xml` requires Java 11 → suggests setting `JAVA_HOME` or updating `maven.compiler.source`.

### "Port 8080 is already in use"

```
> Something is using port 8080, find out what
```

The AI calls `trace_port_conflict(8080)` → finds a leftover Tomcat process from yesterday → offers `taskkill /PID 12456 /F`.

### "My Docker backend silently crashes"

```
> Check the logs for my-api container
```

The AI calls `docker_container_logs("my-api")` → finds `java.sql.SQLException: Connection refused` → suggests checking the database container is running.

### "Is my environment set up correctly for this project?"

```
> Diagnose C:\Projects\my-dotnet-app
```

The AI calls `diagnose_project` → finds no `bin/obj` (never built), `.env.example` exists but no `.env`, dotnet SDK 6 installed but project targets `net8.0` → gives step-by-step fix.

### 7. `run_and_analyze`

Runs a build/start command, captures its output, and auto-analyzes any errors. Combines command execution with stack trace analysis in one step — no need to copy-paste errors.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `command` | string | ✅ | — | The command to run (e.g. `mvn clean install`, `python app.py`, `dotnet build`) |
| `cwd` | string | ❌ | `""` | Working directory for the command |
| `timeout` | integer | ❌ | `120` | Max seconds to wait for the command to finish |
| `tail_lines` | integer | ❌ | `200` | Number of output lines to include in the response |

**What it does:**

1. Executes the command in a subprocess (with approval via MCP tool-call confirmation)
2. Captures both stdout and stderr
3. If the command **fails** (non-zero exit code), automatically runs the stack trace analyzer
4. Returns the full output plus structured root cause and fix suggestions

**Example — Build failure:**

```json
// Input: command="mvn clean install", cwd="C:\\Projects\\my-api"
{
  "command": "mvn clean install",
  "cwd": "C:\\Projects\\my-api",
  "exit_code": 1,
  "success": false,
  "elapsed_seconds": 14.32,
  "total_output_lines": 245,
  "output": "...",
  "note": "Output trimmed to last 200 of 245 lines.",
  "analysis": {
    "detected_language": "java",
    "exception_type": "java.lang.NullPointerException",
    "error_message": null,
    "key_stack_frames": [
      {"method": "com.example.MyService.process", "file": "MyService.java", "line": 42}
    ],
    "diagnoses": [
      {
        "root_cause": "Null reference accessed",
        "suggested_fix": "Check the line indicated in the stack trace. Add null checks or use Optional."
      }
    ],
    "primary_root_cause": "Null reference accessed",
    "primary_fix": "Check the line indicated in the stack trace. Add null checks or use Optional."
  }
}
```

**Example — Successful build:**

```json
// Input: command="mvn clean install -DskipTests", cwd="C:\\Projects\\my-api"
{
  "command": "mvn clean install -DskipTests",
  "cwd": "C:\\Projects\\my-api",
  "exit_code": 0,
  "success": true,
  "elapsed_seconds": 8.71,
  "total_output_lines": 120,
  "output": "..."
}
```

---

### "I got a stack trace in my run window — what went wrong?"

```
> Analyze this error: java.lang.ClassNotFoundException: com.example.MyService
```

The AI calls `analyze_stacktrace` → detects Java, identifies the missing class → suggests checking pom.xml dependencies and classpath.

### "My app crashed — check the log file"

```
> Analyze the errors in C:\Projects\my-api\logs\app.log
```

The AI calls `analyze_stacktrace(log_file=...)` → reads last 100 lines → finds `ModuleNotFoundError: No module named 'requests'` → suggests `pip install requests`.

### "Build my project and tell me what's wrong"

```
> Run mvn clean install in C:\Projects\my-api and fix any errors
```

The AI calls `run_and_analyze("mvn clean install", cwd="C:\\Projects\\my-api")` → captures all output → detects `NullPointerException` at `MyService.java:42` → suggests null check fix.

---

## Architecture

```
doctor-context-mcp/
├── src/
│   ├── __init__.py     # Package marker
│   ├── __main__.py     # Entry point (python -m src)
│   └── server.py       # All tools — single-file MCP server
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # This file
```

The server is intentionally a single file with no external dependencies beyond `mcp`. All detection and auditing is done via:

- **Subprocess calls** — `java -version`, `dotnet --version`, `netstat`, `docker`, etc.
- **File parsing** — `pom.xml` (XML), `build.gradle` (regex), `requirements.txt` (line parsing), `.csproj` (XML)
- **Environment inspection** — `os.environ` for `JAVA_HOME`, `VIRTUAL_ENV`, etc.

---

## Platform Support

Currently optimised for **Windows**. Port conflict tracing uses `netstat -ano` and `tasklist`. For Linux/macOS support, the `trace_port_conflict` tool would need `lsof -i :{port}` as an alternative.

All other tools (audit, diagnose, Docker, env vars, stack trace analysis) work cross-platform.
