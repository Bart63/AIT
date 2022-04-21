import matplotlib.pyplot as plt
import numpy as np

# Use 0.15

def generate_sigmoid(exponent:float, max_val:float, min_val: float):
    return lambda x: max_val - (
        (max_val - min_val) * x / (1 + x**exponent) ** (1/exponent) 
    ) 

if __name__ == '__main__':
    x = np.linspace(0, 2000, 500)
    exponent = 0.5
    sigmoid = generate_sigmoid(exponent, 6, 0.8)

    y = sigmoid(x)

    plt.scatter(x, y, s=10)
    plt.title('Zmodyfikowana funkcja sigmoid.')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('y')
    plt.show()
    plt.clf()

    print(f'Exponent {exponent}')
    print(sigmoid(1), sigmoid(2), sigmoid(5), sigmoid(10), sigmoid(20), sigmoid(1000))
    print()