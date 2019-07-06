#TODOs:

1. Functional space stats logging

Look into the problem in terms of functional space i.e. take {x,net(x)) pairs to be single vectors and measure and denote the differences between the mappings made by the neural network at different Ws

2. Two way connectedness - towards n-ary connections

Examine connectivity between all minima considered to be lower than equal to target loss instead of simply jumping to lowest minima being the target. This way, there will be a generalized algorithm that produces minima with clearly defined paths between them.  

3. Weighting loss towards connectivity
loss = cls_loss + 0.00001 * avg_norm(w1-w2) 

4. Recursive partitioning around the center
Let the gradient descent start from (w1+w2)/2
Use loss = cls_loss + gamma * (norm(w1-model) + norm(w2-model))

5. Don't use MNIST. It displays convex behavior wrt neural nets. Extend to CIFAR and others.
