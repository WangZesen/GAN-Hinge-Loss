## GAN with Hinge Loss Implemented with Tensorflow 2.0

An implementation of hinge version of GAN loss from Geometric GAN (https://arxiv.org/pdf/1705.02894.pdf)

### Trained After 0 Epochs
![Epoch 0](../master/samples/000.jpg?raw=true)

### Trained After 10 Epochs
![Epoch 25](../master/samples/010.jpg?raw=true)

### Trained After 50 Epochs
![Epoch 50](../master/samples/050.jpg?raw=true)

### Trained After 100 Epochs
![Epoch 100](../master/samples/100.jpg?raw=true)

### Trained After 200 Epochs
![Epoch 200](../master/samples/200.jpg?raw=true)

### Trained After 400 Epochs
![Epoch 400](../master/samples/400.jpg?raw=true)

Comments:
1. The hinge loss did not oscillate much during the training.
2. It seems to take much time for convergence. It did not converge at the end. (The convergence state is `gen_loss=0` and `dis_loss=2`)

Todo:
- [ ] Plot training loss curve
