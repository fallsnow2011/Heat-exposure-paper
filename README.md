# Heat Exposure Paper

Code and scripts for thermal exposure, shade, connectivity and inequality analyses.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Google Earth Engine JavaScript scripts are intended to run in the Earth Engine Code Editor. Python scripts that call Earth Engine require local authentication.

## Directories

- `pipeline_scripts/`: processing and analysis scripts
- `figure_scripts/main_figures/`: main figure scripts
- `figure_scripts/supplementary/`: supplementary figure scripts
- `verification/`: validation and consistency checks

## Data layout

The scripts expect the following directories at repository root:

```text
Heat-exposure-paper/
|-- pipeline_scripts/
|-- figure_scripts/
|-- verification/
|-- requirements.txt
|-- README.md
|-- GEE_LST_Baseline/
|-- shadow_maps/
|-- city_boundaries/
|-- results/
`-- paper/
    `-- final-SI/
        `-- data/
```

This repository does not bundle raw geospatial inputs or large intermediate rasters.
