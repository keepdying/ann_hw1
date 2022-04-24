import numpy as np
from tqdm import tqdm

class Perceptron:

    def __init__(self, num_iter=10, const=0.1, verbose=False):
        self.const = const
        self.num_iter = num_iter
        self.errors = []
        self.verbose = verbose

    def train(self, X, y, weight=None):
        input_count = X.shape[0]

        if weight is None:
            self.w = np.random.rand((X.shape[1] + 1))
        else: self.w = weight
        self.bias = 1

        for iterr in range(self.num_iter):
            
            error = 0
            for index in tqdm(range(input_count)):
                update = self.const * (y[index] - self.predict(X[index]))
                self.w += update * X[index]
                self.bias += update * 1

                if update != 0.0: error += 1

            self.errors.append(error)

            if self.verbose: print("Epoch {iterr}/{num_iter}, Error = {error}".format(iterr=(iterr+1), num_iter=self.num_iter, error=error))
            else: print("Epoch {iterr}/{num_iter}".format(iterr=(iterr+1), num_iter= self.num_iter))
            
            if iterr > 1:
                if self.errors[iterr] == 0 and self.errors[iterr - 1] == 0:
                    print("Stopping training since last 2 epochs completed with 0 error...")
                    break
        return self

    def predict(self, X):
        return self.__activation(np.dot(X, self.w) + self.bias)

    def __activation(self, x):
        return np.where(x >= 0.0, 1, -1)

def main():
    unit = Perceptron(num_iter= 10, const= 0.1, verbose= True)
    weights = np.array([1, 0], dtype='float64')
    data_x = np.array([[1, -1], [0, 1], [0.5, -1]])
    data_y = np.array([-1, 1, -1])

    unit.train(X= data_x, y= data_y, weight= weights)
    print(unit.w)
if __name__ == "__main__":
    main()