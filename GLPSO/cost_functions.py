import numpy as np


# 1 Sphere
def sphere(variables):
    return np.sum(np.square(variables))


# 2 Ackley
def ackley(variables):
    variable_num = len(variables)
    variables = np.array(variables)
    tmp1 = 20.-20.*np.exp(-0.2*np.sqrt(1./variable_num*np.sum(np.square(variables))))
    tmp2 = np.e-np.exp(1./variable_num*np.sum(np.cos(variables*2.*np.pi)))
    return tmp1+tmp2


# 3 Rosenbrock
def rosenbrock(variables):
    variable_num = len(variables)
    variables = np.array(variables)
    f = 0
    for i in range(variable_num-1):
        f += 100*np.power(variables[i+1]-np.power(variables[i],2),2)+np.power(variables[i]-1,2)
    return f


# 4 SumOfDifferentPower
def sum_different_power(variables):
    variable_num = len(variables)
    variables = np.array(variables)
    tmp = 0
    for i in range(variable_num):
        tmp += np.power(np.absolute(variables[i]),i+2)
    return tmp


# 5 Griewank
def griewank(variables):
    variable_num = len(variables)
    variables = np.array(variables)
    tmp1 = 0
    tmp2 = 1
    for i in range(variable_num):
        tmp1 += np.power(variables[i],2)
        tmp2 *= np.cos(variables[i]/np.sqrt(i+1))
    return tmp1/4000-tmp2    


# 6 Rastrigin
def rastrigin(variables):
    variable_num = len(variables)
    variables = np.array(variables)
    tmp1 = 10 * variable_num
    tmp2 = 0
    for i in range(variable_num):
        tmp2 += np.power(variables[i],2)-10*np.cos(2*np.pi*variables[i])
    return tmp1+tmp2


# 7 Zakharov
def zakharov(variables):
    variable_num = len(variables)
    variables = np.array(variables)
    tmp1 = 0
    tmp2 = 0
    for i in range(variable_num):
        tmp1 += variables[i] * variables[i]
        tmp2 += (i+1)*0.5*variables[i]
    return -tmp1+np.power(1/2*tmp2,2)+np.power(1/2*tmp2,4)
