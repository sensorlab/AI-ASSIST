# AI-ASSIST

**AI-ASSIST** is a cutting-edge research project dedicated to improving the analysis and control of power grids, with a focus on real-time security assessment and grid stability estimation. The project uses advanced AI-based methods to assess how safe and stable the power grid is under current operating conditions.
 
By analyzing live measurements and comparing them with an extensive historical database, **AI-ASSIST** uses pattern recognition techniques to identify similarities with known operating scenarios. This allows the system to assess potential risks and predict system behavior with high accuracy, providing operators with valuable insights for their decision-making.
 
**AI-ASSIST** is a joint project of [**SensorLab**](https://sensorlab.ijs.si/) at the Jožef Stefan Institute, the [**Laboratory of Electric Power Supply**](https://lpee.fe.uni-lj.si/en/) at the Faculty of Electrical Engineering and [**ELES**](https://www.eles.si/en/), combining expertise in the fields of artificial intelligence, power systems and grid operation.

## 📊 Project Overview

This repository contains code, data, and analyses related to the AI-ASSIST project. Current analyses focus primarily on the IEEE 39-bus test case, with selected results included in this repository. Research on the Slovenian power grid is ongoing and will be incorporated as it progresses.


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

- **Slovenian Grid Network** *(in progress)*  
  Due to data sensitivity, this part of the analysis is excluded from the public repository. Access to the data may be made available upon request in the future.


## 💰 Funding

The AI-ASSIST project receives funding from the Slovenian Research and Innovation Agency (ARIS) under Grant Agreement No. L2-50053.


## 👥 Contributors

- **SensorLab**, Jožef Stefan Institute  
- **Laboratory of  Electric Power Supply**, Faculty of Electrical Engineering, University of Ljubljana
- **ELES**, Slovenian transmission system operator


## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.


## 📫 Contact

For inquiries related to the project, please refer to the contact information on the [Laboratory of Electric Power Supply website](https://lpee.fe.uni-lj.si/en/personnel/urban-rudez-ph-d/) or [SensorLab website](https://sensorlab.ijs.si/about/).
