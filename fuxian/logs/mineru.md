# MinerU Run Log

- **Date**: 2026-06-15
- **Command**: `mineru -p fuxian/paper/paper.pdf -o fuxian/paper_md -b pipeline`
- **MinerU version**: 3.3.1

## Result

**Failed** before PDF processing.

## Error

```
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')
```

MinerU's httpx client does not accept `socks://` from environment proxy settings.

## Workaround attempted

Continue analysis from:

- `fuxian/paper/paper.pdf` (downloaded, ~15MB)
- `README.md`, project page, arXiv metadata
- Source under `source/WorldComposer/WorldComposer/`

## Retry suggestion

```bash
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY http_proxy https_proxy all_proxy
mineru -p fuxian/paper/paper.pdf -o fuxian/paper_md -b pipeline
```

Or fix proxy to `http://127.0.0.1:7897` if MinerU supports it.