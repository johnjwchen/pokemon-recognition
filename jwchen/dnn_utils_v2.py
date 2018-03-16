# coding: utf-8
import minpy.numpy as np

def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	cache = A
	return A.asnumpy(), cache.asnumpy()

def sigmoid_backward(dA, cache):
	A = cache
	dZ = dA * A * (1 - A)
	return dZ.asnumpy()

def relu(Z):
	cache = Z
	A = np.maximum(Z, 0)
	return A.asnumpy(), cache.asnumpy()

def relu_backward(dA, cache):
	Z = cache
	dZ = Z > 0
	return dZ.asnumpy()