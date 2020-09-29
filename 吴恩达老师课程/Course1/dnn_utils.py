import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    activation_cache = Z
    return A, activation_cache

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    A = 1 / (1 + np.exp(-Z))
    # sigmoid 函数的导数
    dA_dZ = A * (1 - A)
    
    # dZ = dJ_dZ = dJ_dA * dA_dZ = dA * dA_dZ
    dZ = dA * dA_dZ
    assert (dZ.shape == Z.shape)

    return dZ

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)
    
    # 方法二（np.where 映射 if 判断，函数作用于每个元素上，而不是整个数组）
    #A = np.where(Z>0, Z, 0)
    #A = np.where(Z<=0, 0, Z)
    
    # 方法三（np.vectorize 映射 if 判断，函数作用于每个元素上，而不是整个数组）
    #func = lambda x: x if x>0 else 0
    #func = lambda x: 0 if x<= 0 else x
    #func_vector = np.vectorize(func)
    #A = func_vector(Z)
    
    # 方法四（逐元素的for 循环中进行判断）
    #A = np.array(Z, copy=True)
    #for i in range(Z.shape[0]):
        #for j in range(Z.shape[1]):
            # 大于 0 就保留原来的值，也可以判断 A[i, j]，因为取值等于 Z[i, j] 
            #if Z[i, j] <= 0:
                #A[i, j] = 0
                
    assert(A.shape == Z.shape)
    activation_cache = Z
    return A, activation_cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    # Z^l  A^l=f(Z^l)  dZ^l  dA^l 的shape都是一样的
    Z = cache
    
    # 计算dZ，方法一（相当于把 dA * dA_dZ 融合在一起了）
    # 如果直接对 A 操作，而不是复制个 dZ 出来，那么函数运行结束后，dA 会被改变！！！！ 但是后面也用不到 dA 了啊？？？？
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    # 计算dZ，方法二（相当于把 if 判断加在数组上，函数作用于每个元素上，而不是整个数组）
    # relu 函数的导数
    #dA_dZ = np.where(Z<=0, 0, 1)
    #dZ = dA * dA_dZ
    
    # 计算dZ，方法三（也相当于把 if 判断加在数组上，函数作用于每个元素上，而不是整个数组）
    #func = lambda x: 0 if x<= 0 else 1
    #func_vector = np.vectorize(func)
    #dA_dZ = func_vector(Z)
    #dZ = dA * dA_dZ
    
    # 计算dZ，方法四（逐元素的for 循环中进行判断）    
    #dZ = np.array(dA, copy=True)
    #for i in range(dZ.shape[0]):
        #for j in range(dZ.shape[1]):
            # Z 中取值大于 0 的话，dZ 中相应位置保留原值
            #if Z[i, j] <= 0:
                #dZ[i, j] = 0
                
    assert (dZ.shape == Z.shape)

    return dZ

def tanh(Z):
    """
    Implements the tanh activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of tanh(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    activation_cache = Z

    return A, activation_cache

def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single TANH unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    A = 1 / (1 + np.exp(-Z))
    
    # tanh 函数的导数
    #dA_dZ = 1 - np.power(A, 2)
    dA_dZ = 1 - A**2
    
    dZ = dA * dA_dZ

    assert (dZ.shape == Z.shape)

    return dZ

def leaky_relu(Z):
    """
    Implement the leaky relu function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    # 方法一
    np.maximum(0.01*Z, Z)
    
    # 方法二（np.where 映射 if 判断，函数作用于每个元素上，而不是整个数组）
    A = np.where(Z<=0, 0.01*Z, Z)
    #A = np.where(Z>0, Z, 0.01*Z)
    
    # 方法三（np.vectorize 映射 if 判断，函数作用于每个元素上，而不是整个数组）
    #func = lambda x: 0.01*x if x <= 0 else x
    #func_vector = np.vectorize(func)
    #A = func_vector(Z)
    
    assert(A.shape == Z.shape)

    activation_cache = Z 
    return A, activation_cache

def leaky_relu_backward(dA, cache):
    """
    Implement the backward propagation for a single Leaky ReLU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    # Z^l  A^l=f(Z^l)  dZ^l  dA^l 的shape都是一样的
    Z = cache
    
    # 会报错的方法，赋的值是二维的
    #dZ = np.array(dA, copy=True)
    #dZ[Z <= 0] = 0.01 * dZ
    
    # 计算dZ，方法一（相当于把 if 判断加在数组上）
    # relu 函数的导数
    dA_dZ = np.where(Z<=0, 0.01, 1)
    dA_dZ = np.where(Z>0, 1, 0.01)
    dZ = dA * dA_dZ
    
    # 计算dZ，方法二（也相当于把 if 判断加在数组上）
    #func = lambda x: 0.01 if x<=0 else 1
    #func = lambda x: 1 if x>0 else 0.01
    #func_vector = np.vectorize(func)
    #dA_dZ = function_vector(Z)
    #dZ = dA * dA_dZ
    
    assert (dZ.shape == Z.shape)

    return dZ