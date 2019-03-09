"""
Script written for CV-Aid Design project at McGill
@author jamestang
@version 1.0
our script attempts to perform fast ICA on sensor data collected by phone

Independent component analysis (ICA) is a statistical and computational technique for revealing 
hidden factors that underlie sets of random variables, measurements, or signals.

ICA defines a generative model for the observed multivariate data, which is typically given as 
a large database of samples. In the model, the data variables are assumed to be linear mixtures of some 
unknown latent variables, and the mixing system is also unknown. The latent variables are assumed nongaussian 
and mutually independent, and they are called the independent components of the observed data. 
These independent components, also called sources or factors, can be found by ICA.

ICA is superficially related to principal component analysis and factor analysis. ICA is a much more powerful technique, 
however, capable of finding the underlying factors or sources when these classic methods fail completely.

The data analyzed by ICA could originate from many different kinds of application fields, including digital images, 
document databases, economic indicators and psychometric measurements. In many cases, the measurements are given as a 
set of parallel signals or time series; the term blind source separation is used to characterize this problem. 
Typical examples are mixtures of simultaneous speech signals that have been picked up by several microphones, 
brain waves recorded by multiple sensors, interfering radio signals arriving at a mobile phone, or parallel time 
series obtained from some industrial process.
reference: https://www.cs.helsinki.fi/u/ahyvarin/whatisica.shtml

"""

from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pandas as pd

DATA_LOAD_PATH = './data/AnalysisData.csv'
DATA_SAVE_PATH = './data/ICA_data.csv'
PARAMS = {}

# setting up hyperparams
PARAMS['n_components'] = None
PARAMS['algorithm'] = 'parallel'
PARAMS['whiten'] = True
PARAMS['fun'] = 'logcosh'
PARAMS['fun_args'] = None
PARAMS['max_iter'] = 2000
PARAMS['tol'] = 0.0001
PARAMS['w_init'] = None
PARAMS['random_state'] = None

# read csv files
X = pd.read_csv(DATA_LOAD_PATH,names = ["IMAGE_IDX", "ACC_X", "ACC_Y", "ACC_Z","MAG_X","MAG_Y","MAG_Z","ROT_VECTOR"])
# print data cahracteristics
print("Data file used: "+DATA_LOAD_PATH)
print("Data dimension: ",X.shape)
# drop the index column
X = X.drop(columns=['IMAGE_IDX'], axis=1)
print(X.head())

"""
Parameters:	
n_components : int, optional
Number of components to use. If none is passed, all are used.

algorithm : {‘parallel’, ‘deflation’}
Apply parallel or deflational algorithm for FastICA.

whiten : boolean, optional
If whiten is false, the data is already considered to be whitened, and no whitening is performed.

fun : string or function, optional. Default: ‘logcosh’
The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’. You can also provide your own function. It should return a tuple containing the value of the function, and of its derivative, in the point. Example:

def my_g(x):
return x ** 3, (3 * x ** 2).mean(axis=-1)

fun_args : dictionary, optional
Arguments to send to the functional form. If empty and if fun=’logcosh’, fun_args will take value {‘alpha’ : 1.0}.

max_iter : int, optional
Maximum number of iterations during fit.

tol : float, optional
Tolerance on update at each iteration.

w_init : None of an (n_components, n_components) ndarray
The mixing matrix to be used to initialize the algorithm.

random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""

print("Hyperparameters for ICA filter")
print(PARAMS)
# create FastICA object
ICA_transformer = FastICA(
	n_components=None,
	algorithm='parallel', 
	whiten=True, 
	fun='logcosh', 
	fun_args=None, 
	max_iter=2000, 
	tol=0.0001, 
	w_init=None, 
	random_state=None)

# transform X
X_transformed = ICA_transformer.fit_transform(X)
X_transformed = pd.DataFrame(data=X_transformed,columns = ["ACC_X", "ACC_Y", "ACC_Z","MAG_X","MAG_Y","MAG_Z","ROT_VECTOR"])
print("Post ICA data chracteristics")
print(X_transformed.shape)
print(X_transformed[:5])

# save to file
print("Saved processed data to: "+DATA_SAVE_PATH)
X_transformed.to_csv(DATA_SAVE_PATH,encoding='utf-8')

