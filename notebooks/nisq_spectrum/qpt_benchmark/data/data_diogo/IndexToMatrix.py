import numpy as np 
import tensorflow as tf

def GetMatricesFromIndex(index,n_qubit):
    '''
        Returns initial and final rotation matrices from index 
        At some point may need to change the order
    '''
    Ry_pi2_m = tf.Variable([[1+0*1j,1+0*1j],[-1+0*1j,1+0*1j]],dtype = tf.complex128)/np.sqrt(2)
    Ry_pi2_p = tf.Variable([[1+0*1j,-1+0*1j],[1+0*1j,1+0*1j]],dtype = tf.complex128)/np.sqrt(2)

    Rx_pi2_p = tf.Variable([[1+0*1j,-1j],[-1j,1]],dtype = tf.complex128)/np.sqrt(2)
    Rx_pi2_m = tf.Variable([[1,1j],[1j,1]],dtype = tf.complex128)/np.sqrt(2)    

    Rx_pi = tf.Variable([[0+0*1j,1],[1,0+0*1j]],dtype = tf.complex128)
    unit_matrix = tf.Variable([[1+0*1j,0+0*1j],[0+0*1j,1+0*1j]],dtype = tf.complex128)     

    rot_down = [Ry_pi2_m,Rx_pi2_p,unit_matrix]
    rot_up = [Ry_pi2_p,Rx_pi2_m,Rx_pi]
    rot = [rot_down,rot_up]
    
    index_a = np.split(np.array(index),n_qubit,axis=0)
    InitRotMatrix = rot[index_a[0][1]][index_a[0][0]] 
    FinRotMatrix = rot[0][index_a[0][2]] 
    for basis_q,up_q,meas_q in index_a[1:]:
        InitRotMatrix = np.kron(InitRotMatrix, rot[up_q][basis_q])
        FinRotMatrix = np.kron(FinRotMatrix, rot[0][meas_q])
    
    return InitRotMatrix, FinRotMatrix