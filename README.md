# AI-ASSIST

**AI-ASSIST** is a research project focused on improving power-grid analysis and control, with a strong emphasis on **real-time security assessment** and **grid stability estimation**.

The project applies advanced AI methods to evaluate how safe and stable a power system is under current operating conditions. By combining live measurements with a large historical database, AI-ASSIST uses pattern-recognition techniques to match current states with known operating scenarios. This enables accurate risk assessment and system-behavior prediction, helping grid operators make informed decisions.

AI-ASSIST is a joint initiative of:

- [**SensorLab**](https://sensorlab.ijs.si/) at the Jožef Stefan Institute
- [**Laboratory of Electric Power Supply**](https://lpee.fe.uni-lj.si/en/) at the Faculty of Electrical Engineering, University of Ljubljana
- [**ELES**](https://www.eles.si/en/), the Slovenian transmission system operator

---

## 📊 Project Overview

This repository contains code, data, and analyses related to the AI-ASSIST project.

At this stage, the public analyses focus primarily on the **IEEE 39-bus test case**, with selected results included in this repository. Research on the Slovenian power grid is ongoing and will be incorporated as appropriate.

---

## 🛠️ Setup and Reproducibility

Use the following steps to set up the environment and reproduce the IEEE 39-bus and ELES pipeline:

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Create a local environment file:

```bash
cp .env.example .env
```

4. Update the variables in `.env` for your environment (API keys, paths, hosts, ports, log verbosity, etc.).

5. Install project dependencies:

```bash
# see Makefile
pip install -e .
```

6. Prepare a dataset:

```bash
uv run ai-assist-prepare <dataset>
```

| Dataset | Description | Source format |
|---|---|---|
| `bus39` | IEEE 39-bus test system | ZIP archive |
| `eles/2026-01` | ELES Slovenian transmission grid, version 2026-01 | ZIP archive |
| `eles/2026-06` | ELES Slovenian transmission grid, version 2026-06 | ZIP archive |
| `interscada/pl` | 41-bus grid (New England 39-bus extended) | Raw CSVs, no archive |
| `interscada/fr` | French transmission grid pilot dataset | Raw CSVs, no archive |

For the two ZIP-based datasets (`bus39`, `eles/2026-01`) this also unpacks the archive into `interim/`; the extracted intermediate files are removed once the ML-ready pickles are built, unless you pass `--no-cleanup`. Set `LOG_LEVEL=DEBUG` (in `.env` or inline, e.g. `LOG_LEVEL=DEBUG uv run ai-assist-prepare bus39`) to see per-file extraction logs and the exact subprocess commands being run.

7. Start the service with Docker Compose:

```bash
docker compose up --build
```

Each dataset is a self-contained directory under `datasets/<dataset>/` holding its raw/interim data alongside its own `prepare.py` + `transform.py`; see `scripts/prepare.py` for how dataset names are discovered and dispatched.

---

## 📁 Repository Structure

```text
├── datasets/               # Public datasets: raw/interim data + prepare.py/transform.py per dataset
│   └── bus39/              # IEEE 39-bus related data and transform.py
├── reports/                # Jupyter notebooks with analyses and reports
├── scripts/                # Shared dataset-prep dispatcher and standalone service scripts
├── src/                    # Source code (metrics, preprocessing, utilities)
└── README.md               # Project documentation
```

---

## 🔬 Analyses Included

- **IEEE 39-Bus System**
  Includes power-flow simulations, machine learning-based stability analysis, and visualization.

- **Slovenian Grid Network** *(in progress)*
  Due to data sensitivity, this part of the analysis is not included in the public repository. Access to related data may be available upon request in the future.

---

## 💰 Funding

The AI-ASSIST project is funded by the **Slovenian Research and Innovation Agency (ARIS)** under **Grant Agreement No. L2-50053**.

---

## 👥 Contributors

- **SensorLab**, Jožef Stefan Institute
- **Laboratory of Electric Power Supply**, Faculty of Electrical Engineering, University of Ljubljana
- **ELES**, Slovenian transmission system operator

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 📫 Contact

For project-related inquiries, please refer to:

- [Laboratory of Electric Power Supply contact page](https://lpee.fe.uni-lj.si/en/personnel/urban-rudez-ph-d/)
- [SensorLab contact page](https://sensorlab.ijs.si/about/)
