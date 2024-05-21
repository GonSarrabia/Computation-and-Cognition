import numpy as np
import random
import matplotlib.pyplot as plt


class ex_2:

    def __init__(self, p_1, X,y_0):
        self.X = X
        self.y_0 = y_0
        self.p_1 = p_1

        self.corr_mat = None
        self.corr_vec = None
        self.weights = None

    def initialization(self):
        self.compute_correlation_matrix()
        self.correlation_vector()
        self.compute_weights_Q_2()


    def compute_correlation_matrix(self):
        self.corr_mat = self.p_1 * np.dot(self.X, self.X.T)

    def correlation_vector(self):
        self.corr_vec = self.p_1 * np.dot(self.X, self.y_0.T)

    def compute_weights_Q_2(self):
        self.weights = np.dot(np.linalg.inv(self.corr_mat), self.corr_vec)



def y_target(weights, X):
    return np.dot(X.T,weights)




def X_init(length, width):
    def y_0_func(x):
        return pow(x, 3) - pow(x, 2)
    X = np.random.uniform(-1, 1, size=(length,width))
    X_without_int = X
    y_0 = y_0_func(X)
    ones_column = np.ones((1, width))
    X = np.concatenate((X, ones_column), axis=0)
    return X, y_0






def Q_3(X, y_0, y):
    plt.figure(figsize=(10, 6))

    plt.scatter(x=X, y=y, label="Our output", color='blue',  alpha=0.7)
    plt.scatter(x=X, y=y_0, label="Wanted output", color='red',  alpha=0.7)

    plt.title('Comparison of y_0 and y_taget')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')

    plt.grid(True)
    plt.legend(loc='best')

    plt.show()


def Q_4(e_1):
    print("Q_4:")
    print("train error:")
    print(mse(y_target(e_1.X, e_1.weights),e_1.y_0))
    print("Generalization error:")
    X_test, y_0_test = X_init(1,500)
    print(mse(y_target(X_test,e_1.weights),y_0_test))



def mse(y_0, y):
    errors = y - y_0
    mse = 0.5*np.mean(np.square(errors))
    return mse





def Q_5():
    print("Q_5:")
    train_error = []
    generalization_error = []
    for P in range(5,105,5):
        t_e = 0
        g_e = 0
        for _ in range(100):
            X, y_0 = X_init(1, P)
            per = ex_2(1 / P, X, y_0)
            per.initialization()
            t_e += mse(y_target(per.X, per.weights), per.y_0)

            X_test, y_0_test = X_init(1, P)
            g_e += mse(y_target(X_test, per.weights), y_0_test)
        train_error.append(t_e/100)
        generalization_error.append(g_e/100)

    print("train_error:")
    print(train_error)
    print("generalization_error:")
    print(generalization_error)
    plot_errors_Q_5(train_error, generalization_error, range(5,105,5) )



def plot_errors_Q_5(train_errors, gen_errors, sample_range):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_range, train_errors, label='Train Error', marker='o')
    plt.plot(sample_range, gen_errors, label='Generalization Error', marker='x')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error')
    plt.title('Train vs Generalization Error')
    plt.legend()
    plt.grid(True)
    plt.show()






def main():
    #Q_1
    length = 1
    width = 500
    X, y_0 = X_init(length,width)
    per = ex_2(1/width, X, y_0)
    per.initialization()


    #Q_2
    print("Q_2:")
    print("correlation matrix:")
    print(per.corr_mat)
    print("correlation vector:")
    print(per.corr_vec)
    print("weight vector:")
    print(per.weights)

    #Q_3
    Q_3(X[:-1, :], per.y_0, y_target(per.X, per.weights))

    #Q_4
    Q_4(per)

    #Q_5
    Q_5()





if __name__ == '__main__':
    main()
