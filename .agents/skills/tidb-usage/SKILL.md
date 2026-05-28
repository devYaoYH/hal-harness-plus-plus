---
name: tidb-usage
description: Reference for connecting to TiDB Cloud (MySQL-compatible) from Python using pymysql.
---

# TiDB Python Usage Reference

## Installation
```bash
pip install pymysql
```

## Connection
```python
import pymysql

conn = pymysql.connect(
    host="gateway01.us-west-2.prod.aws.tidbcloud.com",
    port=4000,
    user="user.root",
    password="password",
    database="hal_traces",
    ssl={"ssl_verify_cert": True, "ssl_verify_identity": True},
    # optional: ssl={"ca": "/path/to/ca.pem"} for custom CA
    autocommit=True,
)
```

TiDB Cloud Serverless uses TLS by default. The system CA store is usually sufficient — only set `ca` if you need a custom cert (e.g. ISRG Root X1 from `https://letsencrypt.org/certs/isrgrootx1.pem`).

## Basic Operations
```python
with conn.cursor() as cur:
    # DDL
    cur.execute("CREATE TABLE IF NOT EXISTS t (id INT PRIMARY KEY, val TEXT)")

    # Insert
    cur.execute("INSERT INTO t (id, val) VALUES (%s, %s)", (1, "hello"))

    # Upsert
    cur.execute(
        "INSERT INTO t (id, val) VALUES (%s, %s) ON DUPLICATE KEY UPDATE val=VALUES(val)",
        (1, "updated"),
    )

    # Select
    cur.execute("SELECT * FROM t WHERE id=%s", (1,))
    row = cur.fetchone()  # tuple or None

    # JSON columns
    import json
    cur.execute("INSERT INTO t2 (id, data) VALUES (%s, %s)", (1, json.dumps({"k": "v"})))

conn.close()
```

## SQLAlchemy (alternative)
```python
from sqlalchemy import create_engine, text

engine = create_engine(
    "mysql+pymysql://user:pass@host:4000/db",
    connect_args={"ssl": {"ssl_verify_cert": True}},
)
with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
```

## Key TiDB notes
- MySQL 8.0 compatible syntax
- Supports JSON column type natively
- `ON DUPLICATE KEY UPDATE` works for upserts
- LONGTEXT supports up to 4GB (good for logs/stdout)
- Always use parameterized queries (%s placeholders) to avoid SQL injection
- Connection pooling: pymysql connections are not thread-safe; use one per thread or use SQLAlchemy pool
