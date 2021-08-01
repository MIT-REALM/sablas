# Safe Control for Black-box Dynamical Systems via Neural Barrier Certificates

## Installation
Clone the repository:
```bash
git clone https://github.com/Zengyi-Qin/adacbf.git
```

Install [PyTorch](https://pytorch.org/)>=1.9. GPU is not required, but recommended for training the ship controller. Then install other dependencies:
```bash
pip install numpy matplotlib tqdm
```

## Testing Pretrained Models

Download `data.zip` from [this link](https://drive.google.com/file/d/1E4SxeWGFFhNPosMGI7NjRtQnYTUaDQDW/view?usp=sharing) and unzip in the main folder. It contains the estimated dynamics of the models and the neural network weights for the controllers and control barrier functions.

### Drone Control
```bash
python test_drone.py --vis 1
```

### Ship Control
Testing in a random environment:
```bash
python test_ship.py --vis 1 --env ship
```
Testing in a river:
```bash
python test_ship.py --vis 1 --env river
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
We use random environments in training. The trained controller can be tested in different environments.