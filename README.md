# Safe Control for Black-box Dynamical Systems with Neural Barrier Certificates

## Installation
Clone the repository:
```bash
git clone https://github.com/Zengyi-Qin/adacbf.git
```

Install [PyTorch](https://pytorch.org/)>=1.9. GPU is not required, but recommended for training the ship controller. Then install other dependencies:
```bash
pip install numpy matplotlib
```

## Testing Pretrained Models

Download `data.zip` from [this link](https://drive.google.com/file/d/1X2b8Voq5xliUYVMDFY6z-_O1bVealk4J/view?usp=sharing) and unzip in the main folder. It contains the estimated dynamics of the models and the neural network weights for the controllers and control barrier functions.

### Drone Control
```bash
python test_drone.py --vis 1
```

### Ship Control
```bash
python test_ship.py --vis 1
```

## Training

### Drone

Since we assume that the system is a black box, we need to first learn the system dynamics from sampled data:
```bash
python sysid_drone.py
```

Then we train the control barrier function and controller:
```bash
python train_drone.py
```

### Ship

First learn the dynamics from sampled data:
```bash
python sysid_ship.py
```

Then train the control barrier function and controller:
```bash
python train_ship.py
```