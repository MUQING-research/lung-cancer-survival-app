# GitHub publishing

The repository-level publishing helper is kept here so deployment and GitHub publishing concerns remain separate from the runtime package. The deployment compatibility alias lives in `04_model_deployment/`.

Preview a GitHub publication from the repository root with:

```bash
python 05_github_publishing/publish_github.py --dry-run
```

Use GitHub Desktop or the repository's normal Git workflow when publishing. The compatibility wrapper can still be used as a deployment alias:

```bash
python 04_model_deployment/deploy_alias.py --check
```
