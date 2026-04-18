"""
Smoke test for Daytona + TiDB integration.

Verifies:
  1. TiDB connection and schema creation
  2. TiDB insert/read round-trip
  3. Daytona sandbox creation, command exec, file upload/download
  4. Cleanup

Usage:
    cd prefect && python smoke_test.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


def test_tidb():
    print("=== TiDB ===")

    from tidb_storage import _connect, ensure_schema

    print("1. Connecting...")
    conn = _connect()
    print(f"   Connected to {conn.host_info}")

    print("2. Creating schema...")
    ensure_schema()

    print("3. Insert/read round-trip...")
    test_key = f"smoke-test-{int(time.time())}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO eval_runs (job_id, task_key, scaffold, model, benchmark, task_id, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (test_key, "t1", "test-scaffold", "test-model", "test-bench", "task-0", "smoke"),
        )
        cur.execute(
            "SELECT scaffold, model, status FROM eval_runs WHERE job_id=%s AND task_key=%s",
            (test_key, "t1"),
        )
        row = cur.fetchone()
        assert row == ("test-scaffold", "test-model", "smoke"), f"Unexpected row: {row}"
        print(f"   Read back: scaffold={row[0]}, model={row[1]}, status={row[2]}")

        cur.execute(
            "INSERT INTO agent_traces (job_id, task_key, agent_id, scaffold, model, benchmark, task_id, run_id, correct) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (test_key, "t1", "test-scaffold__test-model__run1", "test-scaffold", "test-model",
             "test-bench", "task-0", "run1", 1),
        )
        cur.execute(
            "SELECT agent_id, correct FROM agent_traces WHERE job_id=%s AND task_key=%s",
            (test_key, "t1"),
        )
        row = cur.fetchone()
        assert row[0] == "test-scaffold__test-model__run1", f"Unexpected agent_id: {row[0]}"
        assert row[1] == 1, f"Unexpected correct: {row[1]}"
        print(f"   Trace read back: agent_id={row[0]}, correct={row[1]}")

        # Cleanup
        cur.execute("DELETE FROM agent_traces WHERE job_id=%s", (test_key,))
        cur.execute("DELETE FROM eval_runs WHERE job_id=%s", (test_key,))
        print("   Cleaned up test rows.")

    conn.close()
    print("   PASS\n")


def test_daytona():
    print("=== Daytona ===")
    from config_daytona import DAYTONA_API_KEY, DAYTONA_SERVER_URL, DAYTONA_TARGET
    from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxFromImageParams

    print("1. Creating client...")
    client = Daytona(DaytonaConfig(
        api_key=DAYTONA_API_KEY,
        api_url=DAYTONA_SERVER_URL,
        target=DAYTONA_TARGET,
    ))

    print("2. Creating sandbox (python:3.11-slim)...")
    sandbox = client.create(
        CreateSandboxFromImageParams(
            image="python:3.11-slim",
            language="python",
            auto_stop_interval=0,
        ),
        timeout=300,
    )
    print(f"   Sandbox id={sandbox.id}")

    try:
        print("3. Running command...")
        resp = sandbox.process.exec("echo hello && python3 --version")
        assert resp.exit_code == 0, f"Command failed: exit_code={resp.exit_code}"
        output = resp.result.strip() if resp.result else ""
        print(f"   Output: {output}")

        print("4. File upload/download round-trip...")
        test_data = json.dumps({"smoke": True, "ts": time.time()}).encode()
        sandbox.fs.upload_file(test_data, "/tmp/smoke.json")
        downloaded = sandbox.fs.download_file("/tmp/smoke.json")
        assert json.loads(downloaded) == json.loads(test_data), "File content mismatch"
        print(f"   Uploaded and downloaded {len(test_data)} bytes OK")

        print("5. Env var forwarding...")
        sandbox2 = client.create(
            CreateSandboxFromImageParams(
                image="python:3.11-slim",
                language="python",
                env_vars={"SMOKE_TEST_VAR": "works"},
                auto_stop_interval=0,
            ),
            timeout=300,
        )
        try:
            resp2 = sandbox2.process.exec("echo $SMOKE_TEST_VAR")
            env_val = resp2.result.strip() if resp2.result else ""
            assert env_val == "works", f"Env var not forwarded: got '{env_val}'"
            print(f"   SMOKE_TEST_VAR={env_val}")
        finally:
            client.delete(sandbox2)

    finally:
        print("6. Deleting sandbox...")
        client.delete(sandbox)

    print("   PASS\n")


def main():
    failures = []
    for name, fn in [("TiDB", test_tidb), ("Daytona", test_daytona)]:
        try:
            fn()
        except Exception as e:
            print(f"   FAIL: {e}\n")
            failures.append(name)

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")


if __name__ == "__main__":
    main()
