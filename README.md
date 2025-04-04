# AI-ASSIST


**AI-ASSIST** is a research project aimed at enhancing power system analysis and control through AI-assisted methodologies. The project is a collaboration between **SensorLab** at the Jožef Stefan Institute, the **Faculty of Electrical Engineering**, and **ELES**.


## 📊 Project Overview

This repository contains code, data, and analysis related to the AI-ASSIST project. The analyses focus primarily on the IEEE 39-bus test case system, with selected outcomes available here.

> ⚠️ **Note**: Some parts of the analysis involving the Slovenian power grid are confidential and are **not included** in this repository.


## 🛠️ Installation

To set up the project environment and install the necessary dependencies, follow these steps:

1. **Install `transform.py` script dependencies** (optional if you're only exploring data):

```bash
python ./scripts/transform.py
```

2. **(Optional) Install UV**, a fast Python package manager:

```bash
curl -sSf https://astral.sh/uv/install.sh | sh
```

3. Install project dependencies:

```bash
uv pip install -e .
# or without UV
pip install -e .
```


## 📁 Repository Structure

```bash
├── data/                   # Public datasets used for experiments
│   ├── interim/            # Processed data included in the repository
├── reports/                # Jupyter notebooks with analysis and reports
├── scripts/                # Utility scripts
│   └── transform.py        # Data transformation script
├── src/                    # Source code for models and utilities
└── README.md               # Project documentation
```


## 🔬 Analyses Included

- **IEEE 39-Bus System**  
  Includes power flow simulations, machine learning-based stability analysis, and visualization.

- **Slovenian Grid Network** *(Confidential)*  
  Due to the sensitivity of the data, this part of the analysis is excluded from the public repository.


## 🔗 External Links

- 🌐 AI-ASSIST Official Project Page
- 🧪 SensorLab, Jožef Stefan Institute
- 🏫 Faculty of Electrical Engineering, UL


## 💰 Funding

The AI-ASSIST project receives funding from the Slovenian Research and Innovation Agency (ARIS) under Grant Agreement No. L2-50053.


## 👥 Contributors

- **SensorLab**, Jožef Stefan Institute  
- **Faculty of Electrical Engineering**, University of Ljubljana


## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.


## 📫 Contact

For inquiries related to the project, please refer to the contact information on the [official website](https://sensorlab.ijs.si/projects/ai-assist/).
