import numpy as np
import tensorflow as tf

from spam import InitialState, POVM, SPAM
from experimental import generate_pauliInput_circuits, generate_pauli_circuits
from quantum_tools import apply_unitary, measurement, add_shot_noise


def generate_spam_benchmark(n=3, c1=1, c2=1):
    d = 2**n

    init_target = InitialState(d, c=c1)
    povm_target = POVM(d, c=c2)

    spam_target = SPAM(init=init_target, povm=povm_target)

    return spam_target


def generate_spam_data(spam_target, N_spam=None, shots=1024):
    n = int(np.log2(spam_target.d))
    inputs_spam, _ = generate_pauliInput_circuits(n)
    N_spam = inputs_spam.shape[0]

    state = tf.repeat(spam_target.init.init[None, :, :], N_spam, axis=0)
    targets_spam = measurement(state, U_basis=inputs_spam, povm=spam_target.povm.povm)

    # add noise
    if shots is not None:
        targets_spam = add_shot_noise(targets_spam, shots=shots)
    return inputs_spam, targets_spam


def generate_map_data(
    channel_target, spam_target, N_map=None, shots=1024, grid=True, batch_size=1000
):
    n = int(np.log2(channel_target.d))
    inputs_map, _ = generate_pauli_circuits(
        n=n, circuit_target=None, N=N_map, grid=grid
    )
    U_prep, U_basis = inputs_map

    N_map = U_prep.shape[0]

    targets_map = []
    for batch_num in range(N_map // batch_size + 1):
        U_prep_batch = U_prep[batch_num * batch_size : (batch_num + 1) * batch_size]
        U_basis_batch = U_basis[batch_num * batch_size : (batch_num + 1) * batch_size]
        batch_size_current = U_prep_batch.shape[0]

        state = tf.repeat(
            tf.expand_dims(spam_target.init.init, axis=0), batch_size_current, axis=0
        )
        state = apply_unitary(state, U_prep_batch)
        state = channel_target.apply_channel(state)
        probs = measurement(state, U_basis_batch, spam_target.povm.povm)
        if shots is not None:
            probs = add_shot_noise(probs, shots=shots)
        targets_map.append(probs)

    targets_map = tf.concat(targets_map, axis=0)

    return inputs_map, targets_map
