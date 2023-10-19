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