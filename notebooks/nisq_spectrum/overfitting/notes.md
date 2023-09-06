# Overfitting

General question: what characterizes a badly overfitting quantum channel? Since this kind of modelling is both forward and linear, there is no opportunety for things to blow up exponentially with perturbations in paramters or input. Things are manifest smooth and well-behaved.

Initial hypothesis: An overfitted quantum map tends to implement wrong dynamics on states with higher amount of entanglement, compared to lower amounts of entanglement. The cause is that there is a lot of many-body terms in a pauli-string expansion. The few-body terms are few in number, and likely with a high assosiated weight as the map is likely dominated by local effects. However, an overparamteric map might have non-zero weights for a lot of the many-body terms, skewing the dynamics for highly entangled input states. 

Idea for regularization: Penalize weight of many-body terms. Exponentially exposenive if done explicit. Can it be done implicit?


## Richer input and basis adaptation

