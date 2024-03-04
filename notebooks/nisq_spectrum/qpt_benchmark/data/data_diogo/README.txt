Files:
Data.npy ----> contains an array of [Pauli string probabilities, Pauli string indicies] for 
corresponding to 3 different maps

Tgt_Map ----> contains the ground truth vectorized Kraus maps 

IndexToMatrix.py ----> contains a function to map an index into an intial rotation and a final rotation
For each qubit there are 3 indices (a,b,c). 'a' denotes the basis in which to prepare the qubit, 
i.e., $\sigma_{x}$ (a = 0), $\sigma_{y}$ (a = 1), $\sigma_{z}$ (a = 2). 'b' denotes whether
to flip the qubit from '0' to '1'. 'c' denotes the measurement axis and follows the same 
numbering convention as 'a'. For example: (1,0,2) corresponds to preparing $\frac{1}{\sqrt{2}}(\ket{0}-\ket{1})$ 
and measuring along $\sigma_{z}$. (2,0,2) corresponds to preparing $\ket{0}$ and measuring along
$\sigma_z$, i.e., applying no rotations. (1,1,1) corresponds to preparing $\frac{1}{\sqrt{2}}(\ket{0}+i\ket{1})$$ and measuring along
$\sigma_y$.
The leftmost 3 indices correspond to the leftmost qubit and so on. 



