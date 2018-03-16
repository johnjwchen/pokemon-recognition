# coding: utf-8
import minpy.numpy as np

def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	cache = A
	return A, cache

def sigmoid_backward(dA, cache):
	A = cache
	dZ = dA * A * (1 - A)
	return dZ

def relu(Z):
	cache = Z
	A = np.maximum(Z, 0)
	return A, cache

def relu_backward(dA, cache):
	Z = cache
	dZ = Z > 0
	return dZ