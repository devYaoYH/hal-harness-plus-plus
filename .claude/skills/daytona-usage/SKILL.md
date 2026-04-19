---
name: daytona-usage
description: Reference for using the Daytona Python SDK to create, manage, and run code in cloud sandboxes.
---

# Daytona Python SDK Reference

## Installation
```bash
pip install daytona-sdk
```

## Client Setup
```python
from daytona_sdk import Daytona, DaytonaConfig

daytona = Daytona()  # reads DAYTONA_API_KEY from env
# or explicit:
daytona = Daytona(DaytonaConfig(
    api_key="...",
    api_url="https://app.daytona.io/api",  # default
    target="us",  # region
))
```

## Sandbox Lifecycle

### Create from image
```python
from daytona_sdk import CreateSandboxFromImageParams

sandbox = daytona.create(
    CreateSandboxFromImageParams(
        image="python:3.11-slim",  # or Image object
        language="python",
        env_vars={"KEY": "val"},
        auto_stop_interval=0,  # minutes, 0=never
    ),
    timeout=300,  # seconds
)
```

### Create from snapshot
```python
from daytona_sdk import CreateSandboxFromSnapshotParams
sandbox = daytona.create(CreateSandboxFromSnapshotParams(snapshot="snap-id"))
```

### Other lifecycle methods
```python
daytona.get("sandbox-id-or-name")  # -> Sandbox
daytona.list(labels={"env": "prod"})  # -> PaginatedSandboxes
daytona.start(sandbox)
daytona.stop(sandbox)
daytona.delete(sandbox)
```

## Running Commands
```python
response = sandbox.process.exec(
    command="pip install -e . && python main.py",
    cwd="/home/daytona/project",  # optional
    env={"DEBUG": "1"},           # optional
    timeout=1800,                 # seconds, optional
)
response.exit_code    # int
response.result       # stdout string
response.artifacts.stdout  # also stdout
```

## File System
```python
# Upload bytes — positional args: (src, dst)
sandbox.fs.upload_file(b"content", "/home/daytona/file.txt")

# Upload local file path
sandbox.fs.upload_file("/local/file.zip", "/home/daytona/file.zip")

# Download to bytes — positional arg: (path)
data: bytes = sandbox.fs.download_file("/home/daytona/output.json")

# Batch upload
from daytona_sdk import FileUpload
sandbox.fs.upload_files([FileUpload(source=b"data", destination="/home/daytona/f.txt")])

# Directory operations
sandbox.fs.create_folder(path="/home/daytona/newdir", mode="0755")
sandbox.fs.delete_file(path="/home/daytona/old", is_dir=True)
sandbox.fs.list_files(path="/home/daytona/")
sandbox.fs.search_files(path="/home/daytona/", pattern="*.json")
```

## CreateSandboxBaseParams fields
- `name`, `language`, `os_user`
- `env_vars: dict[str, str]`
- `labels: dict[str, str]`
- `public: bool`
- `timeout: int` (minutes)
- `auto_stop_interval: int` (minutes, 0=never)
- `auto_archive_interval: int` (minutes)
- `auto_delete_interval: int` (minutes)
- `volumes`, `network_block_all`, `network_allow_list`, `ephemeral`
