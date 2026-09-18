# Shiny runtime package

This directory is the complete deployment boundary for the NSCLC survival Shiny app. It contains the entry point, runtime modules, precomputed bundle, runtime dependencies, stylesheet, optional visitor-map data, and the small cleaning/preprocessing support modules needed only if the bundle must be rebuilt at startup.

Deploy with:

```bash
python 04_model_deployment/deploy.py --check
python 04_model_deployment/deploy.py
```

The deployment helper uploads only the files listed in `RUNTIME_FILES`. Training code, model tests, documentation, and GitHub helpers are outside this directory and are not uploaded.

The local visitor map uses Natural Earth 1:50m Admin 0 country boundaries from https://github.com/nvkelso/natural-earth-vector.
