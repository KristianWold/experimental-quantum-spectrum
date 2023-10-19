# Academic Writing, Weaknesses

Background Literature Check, Writing in the correct context.

Too much information, when do you have a sufficient overview?

Practice: 
-Read, summarize and archive papers in a structured manner
-Start writing, especially introduction, and "fill in the blanks". Discover blindspots thru writing, and start further research from there.

In the moment of failure, I tell myself:

Stupid, I am naive, lazy.
Oppurtunity for improvement!
Unreflected. 

Stream of councioussness writing!
Quantum channels are good models for describing, mathematically, dynamical processes induced by physical systems we believe fit a quantum description. Quantum channels are functions that map quantum states to quantum states. In order to be physical, they must have certain mathematical constraints. One of them is trace preservation. This comes from a physical motimativon that probability must be conserverd. The probability for any outcome is always unity. Other cases is not unifyable with reality. The other constraint is positivity. This is a refelction of the fact the porbabilities must be positive. Negative probabilities have no well defined meaning. In addtition, quantum channels must be completely positive. This means that the total effect of the quantum channel acting on some subsystem, togehter with an arbitrary observer system that may or may not be correlated, quantumly or clasically, to the subsystem, must also be positive.


Here is a problem:

How does one retrieve an accurate CPTP description of an NISQ circuit from a minimal amount of experimental data?

This is an interesting problem because:
NISQ circuits are noisy and imperfect with respect to their ideal counterpart. In practice, quantum computing is not able to provide an advantage with respect to classical computing in the presence of noise. Obtaining accurate descriptions of the noise of NISQ hardware can help us understand the details of the noise, and help mitigate it.

A general kraus map can be generated in a parametric fashion using QR decomposition. This can be done in a differentiable way, allowing for automatic optimization with libraries like TensorFlow.

Other work tend to optimize models using black-box optimizers from scipy, which tend to get stuck in local minima. 


Experimental results:
Discover characteristics of real hardware at a bigger scale than before. Compare real world results with theoretical toy models. Understand how they are similar, and how they are different. 

Very general methodology agnostic with respect to the spesific hardware. 

High performance implementation, free parallelization and optimal implementations with preexisitng frameworks such as TensorFlow


# Quantum Process Tomography

Quantum Process Tomography is a process of retrieving a description of a quantum map from measurement data. Conventionally, this can be done by performing quantum state tomography on the choi state of a quantum map. The choi state is obtained by applying the quantum map to one subsystem of a maximally entangled pair of substates. In general, quantum process tomography required 3^(2n) measurements in order to retrieve the quantum map, which is quickly becoming intractable for moderatly large quantum systems. 

Today, quantum process tomography is performed more sample efficiently by circumventing tomography of the choi state. Instead, the quantum map is obtained by fittig a functional form directly to data. Usually, some parametric CPTP function is suggested and fitted 

# Fleeing from the Land

Night had barely fallen when the men walked to their boats. They knew they could not stay anymore, for they were no longer men of the country. They knew, all too well, what would fall upon men who slayed their King. Their "King"? Not by any streach of the imagination had the king ever been an idol worth warshipping, for he was cruel and unjust, and the Gods knew what unbespoken devilry he had acted upon his people of Thorgoras. 

The boats could barely fit all the men. 

# Goal

Start writing introduction to Quantum Process Tomography paper, and identiy holes in understanding relating to the context of the study. 

Write 200 (?) words introduction, pointing out every time it seems appropriate to research the background literature more to get concrete details right. Make a complete and self contained introduction that sets the context right, to a very course first (zeroth ?) approximation.

Must be complete by the end of session.

# Introduction Draft

Quantum Computing enables the implementation of several algorithms which offer significant speedups with respect to conventional computing. Popular examples are Shor's algorithm, which enables factorization of integer numbers in polynomial time, and Grover search, which enables search in time proportional to the square-root of the number of elements. However, the successful implementation of such algorithms require a large amount of fault-tolerant qubits, meaning errors introduced by applying quantum gates can be corrected faster than they are introduced. Todays hardware, so-called noisy intermediate-scale quantum (NISQ) hardware, is characterized by having few and noisy qubits which are unfit for quantum error correction. This makes them ultimately unfit for typical quantum algorithms such as Shor's algorithms and Grover's search (kilder på error correction, fault tolerant).

NISQ hardware are unable to implement near ideal unitary transformations given by some desired circuit descriptions. Instead, the effective dynamics are characterized by, among others things, dissipative noise induced by the surrounding environment (kilde). The details of such noise are not generally understood for a broad collection of hardware, which demands methods for obtaining accurate mathematical descriptions for the actual dynamics that NISQ circuits. A typical such method is quantum process tomography (QPT), which is obtained by performing quantum state tomography (QST) on the choi state that result from the NISQ circuit. However, this requires the preparation of a state maximally entangled with a register with ancilla qubits, which is not suitable on small and noisy hardware. Further, retrieval of the n qubit map requires 3^(2n) measurements, which is intractable for moderate n. Today, a popular alternative to tomography of the choi matrix is optimization of a general CPTP map with respect to measurement data. This relieves the need to use ancilla qubits. Also, under the assumption that the underlying dynamics display a large amount of structure, a very limited amount of data is sufficient to recover a description to a high accuracy.

Some examples of optimization-based QPT is explicit constraint optimization of the elements of the choi matrix, or projected or constrained gradient descent optimization of Kraus Operators. These methods have in common that the physicality of the model must be explicitly imposed during optimization. The explicit constrained optimization and the projected gradient method struggles with getting stuck in local minima, while the constrained gradient descent is computationally expensive with respect to vanilla gradient descent. 

In this work, we propose a way of parameterizing general Kraus Maps and SPAM errors such that the resulting model are manifestly physical. This enables free optimization over the whole range of parameters without the possibility of obtaining unphysical models. This in turn allows for vanilla gradient descent to be used, along with its more efficient varients like the Adam optimizer. With this starting point, the framework is easily implementable in TensorFlow, offering out-of-the-box automatic differentiation of the cost function and GPU parallelization. This enables us to recover quantum maps up to five qubits in a reasonable amount of time, given sufficient measurement data. This is a large leap from the usual one to two qubit benchmarks present in current work.

# Consciousness 

What is consciousness? It seems that, indeed, that there is no ultimate truth in the world that can be known for sure. Everything and anything could be subject to delusion. The is no way of providing an ultimate check to see if any of our perception or understanding of the world actually match some ground truth, because no matter what we do, any test we might perform to check the truthness of our believes, we must necessarily use our perception and current understanding of the world. 

Still, there is one, and perhaps one and only truth that seems to be certain. 

# Madness

The men were darting nervous looks between one another when the king gave the order. "Dose the castle in petrol! Burn it all, burn it all!". The king's face contorted in an unseenly manner as he gave the order, with crazy eyes and a miniscule amount of foam around his mouth. Panic started to erupt between the guests as the first few soldiers, very hesitantly, started to perform the order. Glass urns of green oil, which were lined up by the entrance, was being carried around and smashed in various places in the room. The is clear, that the king had been consumed by his own madness.

# Quantum Algorithms Discovery with Reinforcement Learning.

How can we leverage Reinforcement Learning to discover new and improved quantum algorithms without expert knowledge in quantum computing?

# Problem
Citation network, biological network, social data. Natural network stracture.

How can we make a general algorithm for ranking a network?

Classical ranking exist, like page rank. Computationally hard, only cares about incoming links. Can we train neural network to do ranking? What is the most 

## What is the problem?

How can we use machine learning in order to discover and compute the a useful ranking of the nodes of a network?

## What makes it an important problem?

A lot of data has a natural network structure, like biological networks, citation data and social data. An often interesting property of the network to compute is ranking, that is, a numerical value attached to each node indicating its relative "importance" to the network.

A typical ranking algorithm is the page rank algorithm. However, it only cares about incoming links, potentially understating the importance of some nodes, and overstating of others. Also, it is computationally heavy.


## What is your refutable hypothesis
It is possible to leverage machine learning for computing the ranking of a network in a data-driven fashion. Such methods can also be more efficient than classical methods, like page rank.

## What is your contribution
More accurate ranking of networks, and more efficiency.

## What results would demonstrate that you refuted the null hypothesis.
Better scaling laws for machine learning method than classical ranking algorithms. Also, for a given context defining "importance" of a node, machine learning discovering a more accurate ranking than preexisting algorithms would also refute the null hypothesis.

## What methods or data you need to achieve these results?
Classical methods for ranking to benchmark against. Some machine learning driven algorithms, perhaps graph neural network as a starting point. Data needed could be various network like data from various topics.

## What strengths/experience/expertise is needed to attack this problem?

Knowledge in machine learning, experience with data with network structure, graph neural networks, classical algorithms in ranking networks.

## What opportunities makes it timely to solve this problem NOW?

Large-scale hardware for machine learning, like GPU farms, makes it very accessible. Big data is also plenty, especially data with obvious network structures, like social network data. 

## What is the title?
"Learning a network ranking algorithms"
  
## Comments

- The author showed to have a good understanding of the problem.
- More thorough study on the classical methods is suggested to find out the major issues or challenges in the literature.
- It is also suggested to get better understanding of benchmark datasets for the problem statement. 
- Justify why this particular problem is significant to be studied with respect to real-world applications, for instance: how it can help in biological networks or social network analysis?


## Revision

Lets say we have a specific data set with a obvious network structure, for example a social network. We could define a ranking criterion such that we want to find the node (person) which is the "person who influences the dynamics of the network the most". Page rank could provide, where incoming link is interactions between the people, could be a good proxy: the person most people interact with is likely also the person with the most influence of the network. However, this is not always the case. Maybe the the person that influences the network the most is not the one that most people talk to, but the person which communicates the most important information. In this case, page rank would be a bad metric for "most important person of the network". 

With machine learning, maybe it is possible to learn a more suitable criterion for finding the node that influences the network the most.