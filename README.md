# Slope-Matrix-Graph Algorithm Implementation

This repository contains the implementation of a novel **Slope-Matrix-Graph (SMG) algorithm** for analyzing
compositional microbiome data, based on the research published in _Microorganisms_
journal [[1]](https://www.mdpi.com/2076-2607/12/9/1866).

## Overview

The SMG algorithm is designed to identify microbiome correlations primarily based on slope-based distance calculations.
This implementation provides tools for:

- **Differential Analysis**: Compare two datasets and identify objects with significant changes
- **Clustering Analysis**: Group similar objects based on slope distance measurements
- **Network Visualization**: Generate colored Excel outputs showing cluster relationships

## Requirements

- Python 3.12.7
- Required packages: `numpy`, `pandas`, `openpyxl`, `numba`, `llvmlite`

## Installation

``` bash
# Clone the repository
git clone https://github.com/StevenGerrard8/SMG_Code_Base
cd SMG_Code_Base

# Install dependencies (if using virtualenv)
pip install -r requirements.txt
```

## Usage

### 1. Differential Analysis () `DAA-01-29.py`

Compare two datasets to identify differentially abundant objects:

``` bash
python DAA-01-29.py -in1 data1.csv -in2 data2.csv -o results.csv [options]
```

**Parameters:**

- `-in1`: First input CSV file path
- `-in2`: Second input CSV file path
- `-o`: Output CSV file path
- `-pos1`: Positive correlation threshold 1 (default: auto-calculated)
- `-pos2`: Positive correlation threshold 2 (default: auto-calculated)
- `-start`: Start filter value (default: 0.95)
- `-jump`: Filter increment value (default: 0.005)

### 2. Clustering Analysis () `cluster-01-29.py`

Perform clustering analysis and generate network visualization:

``` bash
python cluster-01-29.py -i input.csv -o output.xlsx [-t threshold]
```

**Parameters:**

- `-i`: Input CSV file path
- `-o`: Output Excel file path
- `-t`: Custom threshold (optional, default: auto-calculated)

## Input Data Format

Input CSV files should follow this format:

``` 
ObjectName1,value1,value2,value3,...
ObjectName2,value1,value2,value3,...
ObjectName3,value1,value2,value3,...
...
```

**Requirements:**

- First column: Object identifiers (no blanks or duplicates)
- Remaining columns: Numerical values
- For differential analysis: Both files must have same objects in same order



## Output

### Differential Analysis Output

- CSV file with filtered results for multiple threshold values
- Objects ranked by degree centrality scores
- Edge count information for each filter level

### Clustering Output

- Excel file with distance matrix
- Color-coded clusters in the visualization
- Console output showing cluster statistics

## Performance Optimization

- **Numba JIT**: Critical functions compiled for speed
- **Parallel Processing**: Multi-core CPU utilization
- **Memory Efficiency**: Optimized data structures
- **Union-Find**: Efficient clustering algorithm

## Citation

If you use this implementation in your research, please cite the original paper:

> [Author names]. "A Novel Slope-Matrix-Graph Algorithm to Analyze Compositional..." _Microorganisms_ 12, no. 9 (2024):
1866. [https://doi.org/10.3390/microorganisms12091866](https://doi.org/10.3390/microorganisms12091866)
>

## License

Please refer to the original paper and repository license for usage terms.

## Support

For questions or issues, please refer to the original research paper [[1]](https://www.mdpi.com/2076-2607/12/9/1866) or
create an issue in this repository.
