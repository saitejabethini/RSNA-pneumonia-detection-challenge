import numpy as np

def score(x, W_b):
    W, b = W_b
    return (W.dot(x) + b).A1

def L_i(x, y, W_b):
    scores = score(x, W_b)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss = np.sum(margins)
    return loss

W = np.matrix([[0.2, -0.5, 0.1, 2], [1.5, 1.3, 2.1, 0], [0, 0.25, 0.2, -0.3]])
b = np.array([1.1, 3.3, -1.2])

parameters = (W, b)

cat = np.array([5.3, 6.6, -7, 2.5])
car = np.array([-18.4, 18.6, 2.4, 6.5])
frog = np.array([7.4, -17.7, 5.3, -4.9])

print('Cat: {}\t Loss = {}'.format(score(cat, parameters),
                                   L_i(cat, 0, parameters)))
print('Car: {}\t\t Loss = {}'.format(score(car, parameters),
                                     L_i(car, 1, parameters)))
print('Frog: {}\t Loss = {}'.format(score(frog, parameters),
                                    L_i(frog, 2, parameters)))

'''
$ python3 loss-function.py
Cat: [ 3.16  5.13 -1.7 ]	 Loss = 2.969999999999998
Car: [1.36 4.92 1.98]		 Loss = 0
Frog: [ 2.16   2.52  -3.095]	 Loss = 12.870000000000001
'''
