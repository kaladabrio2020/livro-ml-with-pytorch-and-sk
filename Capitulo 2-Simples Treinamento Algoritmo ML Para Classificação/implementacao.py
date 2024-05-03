import numpy as np

class Perceptron:
    """Perceptron Classificador
    
    Parametros
    ------------
    eta:float
        * Taxa de aprendizado (entre 0.0 e 1.0)
    n_iter:int
        * Número de iteração
    random_state:int
        * Número aleatório com seed para inicialização aleatoria dos pesos

    Atributos
    -----------
    w_:1d_array
        * Pesos  depois do treinamento
    b_:1d-array
        * Bias(vies) depois do treinamento
    erros:list
        * Número de classificações erradas a cada época
    """
    w_ = b_ = None
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Ajuste de treinameto 
        Parametros
        -------------
        X:{array-like}, shape = [n_instancias, n_features|caracteristicas]
        y:{array-like}, shape = [n_instancias]
            * target valores
        """

        rgen = np.random.RandomState(seed=self.random_state)

        #Inicializando pesos aleatoriamente
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)


        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update  = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update

                errors += int(update!=0.0)
            self.errors_.append(errors)
            
        return self  

    def net_input(self, X):
        """Saída liguida"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Retorna a classe rotulada depois de unidade de passo"""
        return np.where(self.net_input(X)>=0.0, 1, 0)
    

import seaborn as sea
import matplotlib.pyplot as plt
from   matplotlib.colors import ListedColormap


def fronteira_de_decisao(X, y, classifier, resolution=0.02, color='pastel'):

    markers = [
    "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*",
    "h", "H", "+", "x", "X", "D", "d", "|", "_",".", ",", 
    ]
    colors = sea.color_palette(color, n_colors=len(np.unique(y)), as_cmap=True) 

    # Plot fronteira de decisão
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid( np.arange(x1_min, x1_max, resolution) , 
                            np.arange(x2_min, x2_max, resolution)
                            )
    
    pred = np.array([xx1.ravel(), xx2.ravel()]).T 
    lab  = classifier.predict(pred)
    lab  =  lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #Plot class
    for idx, cl in enumerate(np.unique(y)):
        dicionario = dict(
            x = X[y==cl, 0],
            y = X[y==cl, 1],
            alpha = 0.8,
            c     = colors[idx],
            marker = markers[idx],
            label = f'Classe {cl}',
            edgecolor = 'black'
        )
        plt.scatter(
            **dicionario
        )


class AdalineGD:
  
    """Adaline LInear Neuron Classificador
    
    Parametros
    ------------
    eta:float
        * Taxa de aprendizado (entre 0.0 e 1.0)
    n_iter:int
        * Número de iteração
    random_state:int
        * Número aleatório com seed para inicialização aleatoria dos pesos
    eta_: str
        * Tipo de taxa de aprendizado ['normal','exponencial']
    Atributos
    -----------
    w_:1d_array
        * Pesos  depois do treinamento
    b_:1d-array
        * Bias(vies) depois do treinamento
    losses_:list
        * Valor de perda a cada época
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1, verbose=False, eta_='normal'):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.eta0 = eta
        self.eta_ = eta_

    def fit(self, X, y):
        random  = np.random.RandomState(seed=self.random_state)
        self.w_ = random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output    = self.activation(net_input)
            errors = (y - output)

            
            self.eta =  self.etaExpo(eta_0=self.eta0 , t=i, s=2) if self.eta_=='exp' \
            else self.eta

            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * np.mean(errors)
            loss = np.mean(errors**2)
            self.losses_.append(loss)
            if self.verbose:
                print(f'Iter : {i} - loss: {loss}')
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        '''Função de ativação linear'''
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.5, 1, 0)
    
    def etaExpo(self, eta_0, t, s=1):
        return eta_0*((0.1)**(t/s))
    


class AdalineSGD:
  
    """Adaline LInear Neuron Classificador
    
    Parametros
    ------------
    eta:float
        * Taxa de aprendizado (entre 0.0 e 1.0)
    
    n_iter:int
        * Número de iteração
    
    shuffle:boll(default:True)
        * None

    random_state:int
        * Número aleatório com seed para inicialização aleatoria dos pesos
    
    eta_: str
        * Tipo de taxa de aprendizado ['normal','exponencial']

    Atributos
    -----------
    w_:1d_array
        * Pesos  depois do treinamento
    b_:1d-array
        * Bias(vies) depois do treinamento
    losses_:list
        * Valor de perda a cada época
    """
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1, verbose=False):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.w_initialized = False
        self.shuffle_ = shuffle

    def fit(self, X, y):
        self._initializer_weights(X.shape[1])
        self.losses_ = []


        for i in range(self.n_iter):
            if self.shuffle_:
                X, y = self.shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_losses = np.mean(losses)
            self.losses_.append(avg_losses)
        return self
    
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initializer_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def shuffle(self, X, y):
        r = self.random.permutation(len(y))
        return X[r], y[r]
    

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        '''Função de ativação linear'''
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.5, 1, 0)
    
    def _initializer_weights(self, m):
        self.random  = np.random.RandomState(seed=self.random_state)
        self.w_ = self.random.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        output    = self.activation(self.net_input(xi))
        errors = (target - output)
        self.w_ += self.eta * 2.0 * xi * (errors)
        self.b_ += self.eta * 2.0 * errors
        loss = errors**2
        return  loss