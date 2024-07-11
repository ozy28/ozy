X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

01.
import numpy as np
02.
 
03.
# sigmoid function
04.
def nonlin(x,deriv=False):
05.
if(deriv==True):
06.
return x*(1-x)
07.
return 1/(1+np.exp(-x))
08.
 
09.
# input dataset
10.
X = np.array([  [0,0,1],
11.
[0,1,1],
12.
[1,0,1],
13.
[1,1,1] ])
14.
 
15.
# output dataset           
16.
y = np.array([[0,0,1,1]]).T
17.
 
18.
# seed random numbers to make calculation
19.
# deterministic (just a good practice)
20.
np.random.seed(1)
21.
 
22.
# initialize weights randomly with mean 0
23.
syn0 = 2*np.random.random((3,1)) - 1
24.
 
25.
for iter in xrange(10000):
26.
 
27.
# forward propagation
28.
l0 = X
29.
l1 = nonlin(np.dot(l0,syn0))
30.
 
31.
# how much did we miss?
32.
l1_error = y - l1
33.
 
34.
# multiply how much we missed by the
35.
# slope of the sigmoid at the values in l1
36.
l1_delta = l1_error * nonlin(l1,True)
37.
 
38.
# update weights
39.
syn0 += np.dot(l0.T,l1_delta)
40.
 
41.
print "Output After Training:"
42.
print l1

01.
import numpy as np
02.
 
03.
def nonlin(x,deriv=False):
04.
if(deriv==True):
05.
return x*(1-x)
06.
 
07.
return 1/(1+np.exp(-x))
08.
 
09.
X = np.array([[0,0,1],
10.
[0,1,1],
11.
[1,0,1],
12.
[1,1,1]])
13.
 
14.
y = np.array([[0],
15.
[1],
16.
[1],
17.
[0]])
18.
 
19.
np.random.seed(1)
20.
 
21.
# randomly initialize our weights with mean 0
22.
syn0 = 2*np.random.random((3,4)) - 1
23.
syn1 = 2*np.random.random((4,1)) - 1
24.
 
25.
for j in xrange(60000):
26.
 
27.
# Feed forward through layers 0, 1, and 2
28.
l0 = X
29.
l1 = nonlin(np.dot(l0,syn0))
30.
l2 = nonlin(np.dot(l1,syn1))
31.
 
32.
# how much did we miss the target value?
33.
l2_error = y - l2
34.
 
35.
if (j% 10000) == 0:
36.
print "Error:" + str(np.mean(np.abs(l2_error)))
37.
 
38.
# in what direction is the target value?
39.
# were we really sure? if so, don't change too much.
40.
l2_delta = l2_error*nonlin(l2,deriv=True)
41.
 
42.
# how much did each l1 value contribute to the l2 error (according to the weights)?
43.
l1_error = l2_delta.dot(syn1.T)
44.
 
45.
# in what direction is the target l1?
46.
# were we really sure? if so, don't change too much.
47.
l1_delta = l1_error * nonlin(l1,deriv=True)
48.
 
49.
syn1 += l1.T.dot(l2_delta)
50.
syn0 += l0.T.dot(l1_delta)
