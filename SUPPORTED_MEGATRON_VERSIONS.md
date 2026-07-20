# Supported Megatron-LM Versions

The automatic patcher is source-layout-specific. A version is listed here only
after the patch has been applied to that exact revision, all modified Python
files have passed AST parsing, and the patch has been reverted cleanly.

| Megatron release | Commit | Source date | Patch ID | Validation |
| --- | --- | --- | --- | --- |
| `core_r0.13.0` | `c550cf6c41c31cd3ec72e05c25ea0c979f2b6631` | 2025-07-25 | `memfrag-c550cf6c-v1` | Apply, AST parse, and revert |

Check the installed tool rather than relying on a local copy of this file:

```bash
megatron-memfrag-patch --list-supported
```

Exact commits are the compatibility boundary. Nearby commits or vendor forks
may work only when every structural anchor still matches, but they are not
considered supported. `--force-version` exists for reviewed experiments; always
inspect `--dry-run` output before applying it.
