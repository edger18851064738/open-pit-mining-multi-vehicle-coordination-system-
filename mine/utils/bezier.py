import numpy as np

class Bernstein:
    """Bernstein多项式基函数实现"""
    def __init__(self, degree, index):
        self.n = degree
        self.i = index
        
    def __call__(self, t):
        """计算Bernstein基函数值"""
        coeff = np.math.comb(self.n, self.i)
        return coeff * (t**self.i) * ((1 - t)**(self.n - self.i))