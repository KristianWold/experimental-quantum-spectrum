# SPAM Benchmarks
Here are notes regarding the SPAM benchmarks on experimental and synthetic data.

## SPAM Convergence

We investigate how well the SPAM models converge to the true SPAM model as a function of initialization and number of training steps. During training, we monitor the fidelity between the model and ground truth inital state and POVM using state fidelity and POVM fidelity, respectively. This lets us clearly see any conclusive signs of overfitting, as this circumvents possible weaknesses of using validation loss as a proxy for overfitting.

We also train multiple spam models on the same ground truth, but with different starting points. This lets us see if the model is able to robustly find the same solution, or if it gets stuck in local minima. This could be particularly interesting to invetigate for the corruption matrix model, since it is more constrained than the general POVM and might get stuck in local minima more easily.

### Results