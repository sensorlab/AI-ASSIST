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

4. Update the variables in `.env` for your environment (API keys, paths, hosts, ports, etc.).

5. Install project dependencies:

```bash
# see Makefile
pip install -e .
```

6. Prepare the dataset in the dataset config directory:

```bash
# required flow: config/<dataset>/dvc.yml
# in this repository, dataset configs are under configs/<dataset>/dvc.yaml
cd configs/bus39
dvc repro
```

7. Start the service with Docker Compose:

```bash
docker compose up --build
```

Pipeline configuration path:

- `config/<dataset>/dvc.yml` (required flow)
- `configs/bus39/dvc.yaml` (path used in this repository)

---

## 📁 Repository Structure

```text
├── configs/                # Configuration files for experiments and pipelines
│   └── bus39/              # DVC pipeline configuration for IEEE 39-bus workflow
├── data/                   # Public datasets used for experiments
│   └── bus39/              # IEEE 39-bus related data
├── reports/                # Jupyter notebooks with analyses and reports
├── scripts/                # Utility and data processing scripts
│   └── bus39/transform.py  # Data transformation for IEEE 39-bus workflow
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
