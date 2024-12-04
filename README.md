# Organic-Molecule-Generation-using-Latent-Diffusion

This repository contains the code and resources for the generation of organic molecules using a **Graph Variational Autoencoder (GraphVAE)** combined with a **Latent Diffusion Model**. The project leverages the **QM9 dataset** to train and evaluate the models.

## Features

- **GraphVAE** for learning latent representations of molecular graphs.
- **Latent Diffusion Model** for generating novel organic molecules.
- Scripts to train, evaluate, and calculate metrics for the models.
- Easily customizable parameters for experimentation.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher.
- `virtualenv` installed (`pip install virtualenv`).

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Organic-Molecule-Generation-using-Latent-Diffusion.git
   cd Organic-Molecule-Generation-using-Latent-Diffusion
   ```

2. Create a virtual environment:
   ```bash
   virtualenv venv
   ```

3. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running Experiments

### 1. Train the GraphVAE
Run the script to train the GraphVAE:
```bash
python script/train_vae.py
```

### 2. Train the Latent Diffusion Model
Once the GraphVAE is trained, use the following script to train the Latent Diffusion Model:
```bash
python script/train_diffusion.py
```

### 3. Evaluate and Calculate Metrics
To calculate the metrics and evaluate the performance of a trained model, use:
```bash
python script/calculate_metrics.py --model_name="<model_name_in_models_folder>"
```

Replace `<model_name_in_models_folder>` with the name of the model file you want to evaluate.

---

## Customizing Parameters

All scripts allow for customization of parameters directly within their respective files. Open the desired script (e.g., `script/train_vae.py`) and modify the relevant sections (e.g., hyperparameters, file paths, or dataset configurations).

For example, in `script/train_vae.py`, you might find a section like this:
```python
# Example parameter block in train_vae.py
learning_rate = 0.001
batch_size = 32
num_epochs = 100
```
Feel free to adjust these values according to your experimentation needs.

---

## Repository Structure

```
Organic-Molecule-Generation-using-Latent-Diffusion/
│
├── data/                    # Dataset and preprocessed data
├── models/                  # Saved models
├── script/                  # Scripts for training, evaluation, and metrics
│   ├── train_vae.py         # Training script for GraphVAE
│   ├── train_diffusion.py   # Training script for Latent Diffusion
│   ├── calculate_metrics.py # Script to calculate metrics and evaluate models
│
├── requirements.txt         # List of Python dependencies
├── README.md                # Project documentation
└── ...
```

---

## Citation

If you use this code in your research, please consider citing:

```
@misc{yourname2024,
  title={Organic Molecule Generation using Latent Diffusion},
  author={Duccio Meconcelli},
  year={2024},
  url={https://github.com/Duccioo/Organic-Molecule-Generation-using-Latent-Diffusion}
}
```

---

Happy experimenting! If you encounter any issues, feel free to open an issue in the repository or contact me directly.
