# [Aligning Relational Learning with Lipschitz Fairness](https://openreview.net/forum?id=ODSgo2m8aE)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Welcome to the official code repository for the **ICLR 2024** paper "[Aligning Relational Learning with Lipschitz Fairness](https://openreview.net/forum?id=ODSgo2m8aE)"! This repository is dedicated to providing a comprehensive and easy-to-use implementation of the methodologies discussed in our paper.

## Introduction
Our work introduces a novel approach to align relational learning with Lipschitz fairness principles. This repository offers a hands-on experience to understand and implement our methods.

## Getting Started

### Prerequisites
- Python 3.x
- Pip package manager

### Installation
To set up the environment and install necessary dependencies, follow these steps:

```shell
git clone https://github.com/chunhuizng/lipschitz-fairness.git
cd lipschitz-fairness
pip install -r requirements.txt
```

## Usage
Our implementation supports various tasks, including node classification and link prediction. Below are the instructions to run each task:

### Node Classification
Navigate to the node classification directory and run the corresponding script:

```shell
cd node/
python jacolip_node.py
```

### Link Prediction
For link prediction tasks, use the following commands:

```shell
cd link/
python jacolip_link.py
```

## Contributing
We welcome contributions and suggestions to improve the code and methodologies. Please feel free to submit issues and pull requests.

## Citation
If you find our work useful in your research, please consider citing:

```
@inproceedings{jia2024aligning,
  title={Aligning Relational Learning with Lipschitz Fairness},
  author={Yaning Jia and Chunhui Zhang and Soroush Vosoughi},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=ODSgo2m8aE}
}
```

## Contact
For any queries regarding the code or research, please contact us at chunhui.zhang.gr@dartmouth.edu.

## Acknowledgments
We would like to thank all the contributors and reviewers who helped in reviewing this research.

## License
This project is licensed under the MIT License - see the LICENSE file for details.