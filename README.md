# Visual Reprogramming with Zeroth-Order Optimization

## Overview
This repository contains the implementation of "Visual Reprogramming with Zeroth-Order Optimization," a research framework designed to enhance large pre-trained models' adaptability to downstream tasks. By utilizing zeroth-order optimization and frequency label mapping, this approach addresses the black-box nature of large models, allowing them to adapt effectively without modifying internal structures.

## Motivation
The increasing use of large models in machine learning, such as GPT or Vision Transformer, brings significant challenges in applying these models to new downstream tasks, especially when data is scarce, and model internals are not accessible. This project proposes a visual prompting method that works without altering the internal layers of a black-box model, providing a flexible and efficient solution to adapt models for new visual classification tasks.

## Key Features
- **Visual Prompting**: Using pre-trained models for new tasks by adding learnable perturbations directly to the input images, without altering the original model architecture.
- **Frequency Label Mapping**: Dynamically generated labels for target tasks, enhancing adaptability and interpretability without random generation.
- **Mixed Optimization**: A hybrid approach combining zeroth-order optimization and traditional first-order methods to balance efficiency and adaptability.
- **Pruning Techniques**: Faster inference and convergence with distributed training and pruning strategies.

## Methodology
The core contribution of this project is the integration of a mixed optimization approach to train visual models for new downstream tasks. This involves:
1. **Visual Reprogramming**: Reprogramming black-box models by applying perturbations to input images.
2. **Frequency-Based Label Mapping**: Improving the semantic connection between source datasets and target downstream tasks.
3. **Mixed Optimization**: Using a combination of first-order and zeroth-order optimizations to maintain model efficiency without direct gradient access.
4. **Distributed Training & Pruning**: Enhancing efficiency through pruning redundant weights and employing distributed training across multiple devices.

## Dataset
The framework was tested on multiple standard datasets in visual modeling:
- **ABIDE**: Brain imaging dataset to assess diagnosis potential.
- **DTD**: Texture dataset for evaluating feature extraction.
- **EuroSAT, Flowers102, Food-101, Stanford Cars, UCF101**: Used for a variety of image classification tasks, assessing generalization on different tasks.
- **Oxford Pets, CIFAR-10, CIFAR-100**: Evaluating adaptability in data-scarce and low-resolution conditions.

## Results
- The framework achieved a 0.5% to 1% accuracy improvement on most datasets compared to previous methods, with a training speed increase ranging from 10% to 25%.
- Experiments demonstrated the feasibility of this approach in various settings, providing better generalization than conventional methods, especially in black-box scenarios.

## Installation
To set up this project locally:
1. Clone the repository:
   ```sh
   git clone https://github.com/SuyuZ1/Visual-Reprogramming.git
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To run experiments, you can modify the parameters in `config.yaml` to specify the dataset, training setup, and optimization method. Use the following command to start the training:
```sh
python train.py --config config.yaml
```

## Acknowledgements
This project was supervised by Dr. Feng Liu at the University of Melbourne. I am also grateful to my family, friends, and peers for their support during this research journey.

## Citation
If you use this work, please cite:
```
@mastersthesis{zhang2024visual,
  title={Visual Reprogramming with Zeroth-Order Optimization},
  author={Zhang, Suyu},
  year={2024},
  school={University of Melbourne}
}
```

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
