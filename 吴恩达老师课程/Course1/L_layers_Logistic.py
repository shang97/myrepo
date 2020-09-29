import numpy as np
import matplotlib.pyplot as plt
#import h5py
import warnings
warnings.filterwarnings('ignore')
    
    
# 多层神经网络的参数初始化
def initialize_parameters(layers_dims):
    """
    初始化所有层的参数，包括输入层和隐藏层之间，隐藏层和输出层之间。
    参数：
        layers_dims - 每层神经元数量的列表，分别为 输入层、隐藏层、输出层
    """
    np.random.seed(3)
    parameters = {}
    
    # 第一个是输入层的神经元数量，所以计算神经网络的层数时要减一
    n_layers = len(layers_dims) - 1 
    
    for l in range(1, n_layers+1):
        # l 对应 layer_dims 的第 l+1 个元素
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1]) 
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        
        # 确保数据的格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))
        
    return parameters


# 不同的激活函数
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def tanh():
    pass


# 单层网络的前向传播，区分不同层
def forward(A_last, W, b, activation):
    """
    根据上一层（l-1）的线性输出A_last 和本层的参数 W,b，计算出本层（l）的线性输出 Z 和激活 A
    返回：
         cache - 元组，包含 元组linear_cache：（上一层的线性输出 A_last，以及本层的参数 W（不需要保存b），
                     和 元素 activation_cache（本层的线性输出 Z）， 用于反向传播
    """
    
    if activation == "sigmoid":
        Z = np.dot(W, A_last) + b
        A = sigmoid(Z)        
    elif activation == "relu":
        Z = np.dot(W, A_last) + b
        A = relu(Z)
    
    assert(A.shape == (W.shape[0], A_last.shape[1]))
    
    linear_cache = (A_last, W, b)
    activation_cache = Z
    cache = (linear_cache, activation_cache)
    
    return A, cache


# 多层网络的前向传播
def L_layers_forward(X, parameters):
    """
    总共 L（n_layers）层，实现输入层（X）到输出层AL（Yhat）的计算
    
    参数：
        parameters - 从W1 到 WL，b1 到 bL（无W0、b0）
    
    返回：
        AL - 最后的激活值
        caches - 缓存列表，共 L 个cache元组：隐藏层的 L-1 个 cache：(linear_cache, activation_cache)，索引从 0 到 L-2
                                            和输出层的一个cache：(linear_cache, activation_cache)，索引为 L-1
    """
    # 参数包括 W和b对，所以要除以2，如果不整除而采用除法的话，结果是float，而range函数要求的输入必须是int
    n_layers = len(parameters) // 2
    
    # 所有隐藏层（1 到 L-1层）的线性输出和 relu激活    
    A_last = A0 = X   
    caches = []
    for l in range(1, n_layers): 
        # 根据上一层的激活 A_last 和本层的参数 Wl 和 bl 来计算本层（l）的激活值 A
        A, cache = forward(A_last, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
        # 本层 A 是计算下一层 A 的 A_last
        A_last = A
    
    # 输出层（第L层）的线性输出和 sigmoid激活
    AL, cache = forward(A_last, parameters['W' + str(n_layers)], parameters['b' + str(n_layers)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))    
    
    return AL, caches


# 计算成本
def compute_cost_2class(AL, Y):
    """
    二分类的交叉熵成本函数。

    参数：
        AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
        Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

    返回：
        cost - 交叉熵成本
    """
    m = Y.shape[1]
    
    # 哈达玛积的矩阵顺序不重要（element-wise product）； axis默认是None，即对所有的元素进行求和
    # print(Y.shape)
    cost = - 1/ m * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y) )
    # print(cost)
    
    # axis默认对所有元素求和，所以当Y是向量时，可以看作矩阵乘法，但是当Y是矩阵（对应多分类）呢？？？
    # I = np.ones((m, 1))
    # cost1 = -1/m * np.dot( (np.log(AL) * Y + np.log(1 - AL) * (1 - Y)), I )       
    # print(np.squeeze(cost1))
    
    # 确保成本是标量
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost


# 反向传播--线性部分
def linear_backward(dZ, linear_cache):
    """为单层实现反向传播的线性部分（第L层）。基于 dZl 计算和输出 dWl、dbl"""

    A_last, W, b = linear_cache 
    # 批数据量
    m = A_last.shape[1]
    
    # 值（向量）都是一样的, 但是互为转置
    dW = np.dot(dZ, A_last.T) / m
    # dW1 = np.dot(A_prev, dZ.T) / m
    
    # db的结果是一致的，注意要keepdims，不要squeeze
    db = np.sum(dZ, axis=1, keepdims=True) / m    
    # I = np.ones((m, 1))
    # db1 = np.dot(dZ, I) / m
    
    # 根据Wl计算dAl-1
    dA_last = np.dot(W.T, dZ)
    
    assert (dA_last.shape == A_last.shape)
    assert (dW.shape == W.shape)
    # 保存b的目的，就是在这里对照一下shape，其他地方完全用不到，但是也是有必要的！！！
    assert (db.shape == b.shape)
    
    return dA_last, dW, db


# 激活函数的求导
def relu_backward(dA, activation_cache): 
    """dZ 的计算也可以写成 Z和激活函数的导数值（Z中的元素如果大于0，导数值为1，其他为0） 的逐元素运算，"""
    Z = activation_cache
    
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    
    # 这个除法是逐元素除法
    S = 1 / (1 + np.exp(-Z))
    dZ = dA * S * (1- S)

    assert (dZ.shape == Z.shape)
    return dZ

def tanh_backward():
    pass


# 单层网络的反向传播，区分不同层
def linear_activation_backward(dA, cache, activation="relu"):
    # （A_last，W）， Z
    linear_cache, activation_cache = cache
     
    # 反向传播到第 l 层神经网络（隐藏层）
    if activation == "relu":
        # 输入dAl，计算dZl
        dZ = relu_backward(dA, activation_cache)
        # 计算 𝑑𝐴_𝑙𝑎𝑠𝑡 和 𝑑𝑊𝑙、𝑑𝑏𝑙
        dA_last, dW, db = linear_backward(dZ, linear_cache)
    
    # 反向传播到第 L 层神经网络（输出层）
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_last, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_last, dW, db


# 多层网络的反向传播
def L_layers_backward(AL, Y, caches):
    """ 从最后一层（第L层）开始，向前反向传播"""
    
    # 列表caches的每个元素是每层的cache元组
    n_layers = L =  len(caches)
    m = AL.shape[1]
    
    # 逐元素除法，和 / 的结果一致
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # dAL1 = - ( Y / AL - (1 - Y) / (1 - AL) )
    
    # 对输出层的反向传播单独计算，传入dAL，计算 dAL-1 和 dWL、dbL
    grads = {}
    # 列表的 index 需要减 1
    current_cache = caches[n_layers - 1]
    grads["dA" + str(n_layers - 1)], grads["dW" + str(n_layers)], grads["db" + str(n_layers)] = linear_activation_backward(dAL, 
                                                                                                    current_cache, "sigmoid")
    
    # 再传播到隐藏层（1 到 L-1层）
    for l in reversed(range(1, n_layers)):
        # 列表的 index 需要减 1
        current_cache = caches[l - 1]        
        # 反向传播到第 l 层，传入 dAl 和 Wl bl Al-1，计算 dZl dWl dbl 和 dAl-1 
        dA_last, dW, db = linear_activation_backward(grads["dA" + str(l)], current_cache, "relu")
        grads["dA" + str(l - 1)] = dA_last
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
    
    return grads


# 多层网络的参数更新
def update_parameters(parameters, grads, learning_rate):
    """梯度下降法更新参数"""
    
    n_layers = L = len(parameters) // 2 # 整除，结果是int而非float
    
    # 参数更新：从 1 层 到 L 层（包含隐藏层和输出层）
    for l in range(1, n_layers + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters


# 多层网络的模型预测
def predict(X, y, parameters):
    """用于预测L层神经网络的结果，当然也包含两层 """
    
    # 批数据量
    m = X.shape[1]
    p = np.zeros((1, m))
    
    #根据参数前向传播，AL是个向量，每个元素对应某个样本的预测概率
    AL, caches = L_layers_forward(X, parameters)
    
    for i in range(0, m):
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    
    print("准确度为: "  + str(float(np.sum((p == y))/m)))
        
    return p


# 多层网络的综合构建
def L_layers_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=True, isPlot=True):
    """实现一个L层神经网络"""
    
    np.random.seed(1)
        
    parameters = initialize_parameters(layers_dims)
    
    costs = []
    # 多层网络的批量训练
    for i in range(num_iterations + 1):    
        AL, caches = L_layers_forward(X, parameters)        
        cost = compute_cost_2class(AL, Y)     
        grads = L_layers_backward(AL, Y, caches) 
        parameters = update_parameters(parameters, grads, learning_rate)
        costs.append(cost)
        if i % 500 == 0 and print_cost:
            print("第", i ,"次迭代，成本值为：", cost)
                
    # 迭代完成，绘制cost变化图
    if isPlot:
        plt.figure(figsize=(6,6))
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters