# 3d-BinPacking_RL-Transformer
Code implementation of the Paper "Solving 3D packing problem using Transformer network and reinforcement learning" (https://www.sciencedirect.com/science/article/abs/pii/S0957417422021716)

## Usage
### Training
To train the model, run the following command in your terminal:

```bash
python train.py --config config.yaml
```

Replace `config.yaml` with your configuration file if needed. You can view additional options by running:

```bash
python train.py --help
```

Note: <br>
Training takes about 4 day to a week to accomplish passable results. 
### Test
To test the model, run the following command in your terminal:

```bash
python test.py --config config.yaml
```

Replace `config.yaml` with your desired configuration file. For more options, use:

```bash
python test.py --help
```