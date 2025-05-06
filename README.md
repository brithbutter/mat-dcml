# Multi-Agent Transformer-based Workload Allocation and Worker Selection (MAT-AS) for Distributed Coded Machine Learning (DCML)

This is the **official implementation** of MAT-AS. The foundational model, Multi-Agent Transformer (MAT), is a novel approach to multi-agent reinforcement learning (MARL) proposed by Wen et al. In this project, MAT is applied to sequentially perform worker selection and workload allocation within a MARL framework. After training, the MAT agent can significantly reduce task completion time and computation cost by learning an efficient policy.

This implementation is based on and adapted from the original MAT repository: https://github.com/PKU-MARL/Multi-Agent-Transformer.

## Features
- **MAT-based agent**: Utilizes the Multi-Agent Transformer (MAT) architecture for coordinated decision-making in MARL.
- **Batch MAT Decision-Making**: Extends the decision making procedure of MAT to acclerate decision-making.
- **Training**: Includes full training scripts with configurable hyperparameters.
- **Benchmark**: Built-in evaluation scripts to assess performance in terms of task completion time and computational cost.

## Installation
To ensure compatibility and prevent issues caused by library version mismatches, this project includes a `requirements.txt` file specifying all dependency versions.

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage
### Training Script
To train the MAT-AS model, run the following command:
``` python
python DCML_MAT_Train.py
```
The DCML simulation environment used is implemented in ```DCML_BID_FIRST_MA_ENV_SingleProcess.py```.
After training, the resulting models will be saved in folder```results/DCML/AS/mat/check/run{}``` replace ```{}``` with the run index.

### Benchmark script
After training, the MAT-AS agent can be evaluated using the benchmark script:
``` python
python DCML_MAT_ALT_Benchmark.py
```
By default, benchmark results are exported as .npy files for further analysis.