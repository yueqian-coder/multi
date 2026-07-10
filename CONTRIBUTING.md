# Contributing

Open an issue for a bug or proposal, or submit a focused pull request.

```bash
python -m pip install -e ".[web,mcp,dev]"
python -m pytest -q
python -m compileall claimscope app.py
```

Keep changes deterministic where possible, add a focused test first, and do not include credentials, private endpoints, or real user data. Clearly label synthetic fixtures. Pull requests should explain user-visible behavior and verification performed.
