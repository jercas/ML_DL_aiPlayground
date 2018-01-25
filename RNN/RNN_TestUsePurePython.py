# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:17:08 2017
    
@author: jercas
"""
# import the compute reliable package
import copy
import numpy as np

np.random.seed(0)

# nonlinearity function
def sigmoid(x):
    """ compute sigmoid nonlinearity """
    output = 1/(1+np.exp(-x))
    return output

# derivative getter function
def sigmoid_output_to_derivative(output):
    """ convert output of sigmoid function to its derivative """
    return output*(1-output)

# training dataset generation
# 声明了一个查找表，该表是一个实数与对应二进制表示的映射
# 二进制将用来表示网路的输入和输出，该查找表帮助将实数转化为其二进制表示
int2binary = {}
# 设置了二进制数的最大长度
binary_dim = 8

# pow(x,y) -- x^y,here 2^8
# 计算了所设置的二进制最大长度对应表示的最大十进制数
largest_number = pow(2,binary_dim)
# 生成十进制转二进制数的查找表
binary = np.unpackbits(
         np.array([range(largest_number)],dtype=np.uint8).T,axis=1
                      )
# 将该查找表复制到 int2binary[] 映射表中
for i in range(largest_number): 
    int2binary[i] = binary[i]
    
# input variables
# learning speed -- 学习速率
alpha = 0.1
# input number -- 输入值；因为训练为二进制加法，一次输入两位字符，需要两个输入
input_dim = 2 
# hidden layers -- 隐藏层大小，回来存储'携带位'
# 注：该数值大小比原理上所需的大很多，可自行尝试调整值，观察对收敛速率(正确率)的影响
#     更高的隐藏层维度会使训练变慢还是变快？更多或更少的迭代次数？
hidden_dim = 16
# round time
n = 1
# output number -- 输出值；预测和的值。
output_dim = 1

# initialize neural network weights
# 各层间连接的"突触"--权值矩阵
# 参照上方参数，理解权值矩阵的结构 n*n
# the synapse between input layer and hidden layer , default shape 2*16
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
# the synapse between hidden layer and output layer , default shape 16*1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
# the synapse between h hidden layers , default shape 16*16
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

# store weights updates between different layers
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# iterate training logic 10000 times
for j in range(10000):
    # random generate a  simple addition problem (a+b=c)
    # random generate the addend (between 0~half of the largest_number)
    # int version of the addend
    a_int = np.random.randint(largest_number/2)
    # use the int2binary inqury the binary encoding of the int number
    a = int2binary[a_int]
    
    # random generate the summand (between 0~half of the largest_number)
    # int version of the summand
    b_int = np.random.randint(largest_number/2)
    # use the int2binary inqury the binary encoding of the int number
    b = int2binary[b_int]
    
    # int version of true answer
    c_int = a_int + b_int
    # use the int2binary inqury the binary encoding of the int number
    c = int2binary[c_int]
    
    # initialize a zeros binary array to store our best guess(binary encoder) -- neural network guessed result
    d = np.zeros_like(c)
    
    # reset the errors to record the convergence condition (gradient descent)
    overallError = 0
    
    # 这两个list会每个时刻不断的记录layer_2的导数值 和 layer_1的值
    layer_2_deltas = list()
    layer_1_values = list()
    # 0时刻是没有之前的隐藏层的，所以初始化一个全为0的隐藏层
    layer_1_values.append(np.zeros(hidden_dim))
    
    # 循环遍历二进制数字
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        # x和图中layer_0一样，x数组中的每个元素包含两个二进制数，一个来自加数a，一个来自被加数b
        # 它通过position变量从a,b中检索，从最右边往左检索
        # 当position=0，检索a最右边一位和b最右边一位
        # 当position=1，左移一位
        x = np.array([[a[binary_dim - position -1],b[binary_dim - position -1]]])
        # 同上，只是把值替换为正确结果
        y = np.array([[c[binary_dim - position -1]]]).T
        
        # hidden layer (input ~+ prev_hidden)
        # np.dot -- the matrix multiplication
        """
        这里就是奥妙所在！一定一定一定要保证你理解这一行！！！
        为了建立隐含层，我们首先做了两件事。
        第一，我们从输入层传播到隐含层（np.dot(X,synapse_0)）,输入x * 权值矩阵synapse_0(即连接输入层和第一个隐藏层的突触)。
        然后，我们从之前的隐含层传播到现在的隐含层（np.dot(prev_layer_1,synapse_h)）。
        在这里，layer_1_values[-1]就是取了最后一个存进去的隐含层，也就是之前的那个隐含层！
        即layer_1_values[-1] == prev_layer_1
        然后我们把两个向量加起来！！！！然后再通过sigmoid函数。
        那么，我们怎么结合之前的隐含层信息与现在的输入呢？当每个都被变量矩阵传播过以后，我们把信息加起来。
        """
        layer_1 = sigmoid(np.dot(x,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
        
        # output layer (input ~+ prev_hidden)
        """
        从隐藏层传播到输出层, 最后一个隐藏层 * 权值矩阵synapse_1(即连接输出层和最后一个隐藏层的突触)
        """
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        
        # did we miss? ... if so by how much?
        # 计算预测误差(预测值和真实值的差),即上图中的亮黄色节点
        layer_2_error = y - layer_2
        # 将导数值存起来，保留每时刻的导数值,即上图中的橘黄色节点
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        # 计算误差的绝对值，并将其相加以获得一个误差的标量，用来衡量传播，最后获得所有二进制位的误差总和
        overallError += np.abs(layer_2_error[0])
        
        # decode estimate so we can print it out
        d[binary_dim - position -1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
        
    # initialize the hidden layer derivative in the next timestep by all zeros np array
    future_layer_1_delta = np.zeros(hidden_dim)
    
    
    """ 
    在上个for循环中，已经完成了所有的正向传播，并计算了输出层的导数，并将其存入一个list中
    接下来进行反向传播，从最后一个时间点开始一直到第一个
    """
    for position in range(binary_dim):
        # 如之前一样，检索输入数据
        # 从左端开始
        x = np.array([[a[position],b[position]]])
        # 末尾出发，从列表中取出当前隐藏层
        layer_1 = layer_1_values[-position-1]
        # 同上，取出前一个隐藏层
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        # 从列表中取出当前输出层的误差
        layer_2_delta = layer_2_deltas[-position-1]
        
        # error at hidden layer
        # 计算当前隐藏层误差
        # 通过当前之后一个时间点 future_layer_1_delta 的误差和当前输出层 layer_2_delta的误差进行计算
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) 
                        + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        
        
        # let's update all our weights so we can try again
        """
        我们已经有了反向传播中当前时刻的导数值，那么就可以生成权值更新的量了（但是还没真正的更新权值）。
        我们会在完成所有的反向传播以后再去真正的更新我们的权值矩阵，这是为什么呢？
        因为我们要用权值矩阵去做反向传播。
        如此以来，在完成所有反向传播以前，我们不能改变权值矩阵中的值。
        """
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += x.T.dot(layer_1_delta)
        # 反向向前传播，将当前隐藏层导数作为前一个隐藏层的导数
        future_layer_1_delta  = layer_1_delta
    
    # 完成反向传播，得到权值要更新的量
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    # 重置update变量
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
        
    # print out progress
    # output logs
    if(j % 1000 == 0):
        print("Round: "+ str(n)) 
        print("Error：" + str(overallError))
        print("Pred" + str(d))
        print("True" + str(c))
        
        out = 0 
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        
        n += 1
        print("------------")