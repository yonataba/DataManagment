# Online Set Selection with Fairness and Diversity Constraints

## Project Overview
This project implements algorithms for online set selection under fairness and diversity constraints. The work is inspired by and builds upon the research presented in the paper:

**"Online Set Selection with Fairness and Diversity Constraints"** by Julia Stoyanovich, Ke Yang, and HV Jagadish ([EDBT 2018](https://dx.doi.org/10.5441/002/edbt.2018.22)).


## Implemented Algorithms
We implemented and experimented with three different online selection algorithms:
1. **Algo1**: A simple diverse top-k selection algorithm that chooses items from a sorted list while maintaining fairness constraints.
2. **Algo2**: An improved version that selects items in an online streaming fashion, using a warm-up phase for score estimation.
3. **Algo3**: A further enhancement that introduces a deferred list mechanism to optimize accuracy while maintaining fairness and diversity constraints.

Each algorithm is designed to maximize a utility score while ensuring that the selection satisfies predefined diversity constraints.

## Code
The main implementation is in `OnlineSetSelection.py`, which contains the core algorithms along with an example experiment using real-world billionaire data.

### Dependencies
The required Python packages are listed in `requirements.txt`.

To install them, run:
```sh
pip install -r requirements.txt
```

### Running the Code
To execute the experiments, simply run:
```sh
python OnlineSetSelection.py
```
This script reads data from `2024 Billionaire List.csv`, processes it using the selection algorithms, and generates visualizations saved in the `plots/` directory.

## Dataset
The dataset used in this project is:
- **2024 Billionaire List** (`2024 Billionaire List.csv`): Contains information about the net worth, gender, and other attributes of the richest individuals in 2024.

## Results
The project generates various plots that illustrate the performance of different algorithms under different warm-up factors. These plots can be found in the `plots/` directory. They visualize:
- The trade-off between walking distance and accuracy.
- The impact of different warm-up factors on selection quality.
- The effects of fairness and diversity constraints on selection accuracy.

## Reference Paper
The theoretical foundation of this project is based on the following paper:

> Julia Stoyanovich, Ke Yang, HV Jagadish, "Online Set Selection with Fairness and Diversity Constraints," EDBT 2018. [Full Paper](https://dx.doi.org/10.5441/002/edbt.2018.22).

## Contributors
This project was developed by:
- Yonatan Baruch Baruch
- Or Saada

