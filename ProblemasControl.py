import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.colors import LightSource
from matplotlib import rcParams
import control as co
import time
import math
import slycot
from control.matlab import *
from numpy import linalg as la
import matlab.engine 
from sklearn.datasets import load_iris

eng=matlab.engine.start_matlab()
eng = matlab.engine.start_matlab("-desktop")
#eng=matlab.engine.connect_matlab('MATLAB_33964')

    
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)

__all__ = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13']


# %% BASIC FUNCTION CLASS
class BasicProblem:
    """
    This is the basic class for a generic optimisation problem.
    """

    def __init__(self, variable_num=2):
        """
        Initialise a problem object using only the dimensionality of its domain.

        :param int variable_num: optional.
            Number of dimensions or variables for the problem domain. The default values is 2 (this is the common option
            for plotting purposes).
        """
        self.variable_num = variable_num
        self.max_search_range = np.array([0] * self.variable_num)
        self.min_search_range = np.array([0] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)
        self.optimal_fitness = 0
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}
        self.plot_object = None
        self.func_name = ''
        self.save_dir = '{0}/function_plots/'.format(os.path.dirname(os.path.abspath(__file__)))

        self.__offset_domain = 0.0
        self.__scale_domain = 1.0
        self.__scale_function = 1.0
        self.__offset_function = 0.0
        self.__noise_type = 'uniform'
        self.__noise_level = 0.0

    def get_features(self, fmt='string', wrd='1', fts=None):
        """
        Return the categorical features of the current function.

        :param str fmt: optional
            Format to deliver the features. Possible options are 'latex' and 'string'. If none of these options are
            chosen, this method returns the equivalent decimal value of the binary sequence corresponding to the
            features, e.g., 010 -> 2. The default is 'string'.
        :param str wrd: optional
            Specification to represent the features. Possible values are 'Yes' (for 'Yes' or 'No'), '1' (for '1' or
            '0'), and 'X' (for 'X' or ' '). If none of these options are chosen, features are represented as binary
            integers, i.e., 1 or 0. The default is '1'.
        :param list fts: optional
            Features to be read. The available features are: 'Continuous', 'Differentiable', 'Separable', 'Scalable',
            'Unimodal', 'Convex' The default is ['Differentiable', 'Separable', 'Unimodal'].

        :return: str or int
        """
        # Default features to deliver
        if fts is None:
            fts = ['Differentiable', 'Separable', 'Unimodal']

        def translate_conditional(value):
            if wrd == 'Yes':
                words = ['Yes', 'No']
            elif wrd == '1':
                words = ['1', '0']
            elif wrd == 'X':
                words = ['X', ' ']
            else:
                words = [1, 0]
            return words[0] if value else words[1]

        # Get the list of features as strings
        features = [translate_conditional(self.features[key]) for key in fts]

        # Return the list according to the format specified
        if fmt == 'latex':
            return " & ".join(features)
        elif fmt == 'string':
            return "".join(features)
        else:
            return sum(features)

    def set_offset_domain(self, value=None):
        """
        Add an offset value for the problem domain, i.e., f(x + offset).

        :param float value:
            The value to add to the variable before evaluate the function. It could be a float or numpy.array. The
            default is None.
        """
        if value:
            self.__offset_domain = value

    def set_offset_function(self, value=None):
        """
        Add an offset value for the problem function, i.e., f(x) + offset

        :param float value:
            The value to add to the function after evaluate it. The default is None.
        """
        if value:
            self.__offset_function = value

    def set_scale_domain(self, value=None):
        """
        Add a scale value for the problem domain, i.e., f(scale * x)

        :param float value:
            The value to add to the variable before evaluate the function. It could be a float or numpy.array. The
            default is None.
        """
        if value:
            self.__scale_domain = value

    def set_scale_function(self, value=None):
        """
        Add a scale value for the problem function, i.e., scale * f(x)

        :param float value:
            The value to add to the function after evaluate it. The default is None.
        """
        if value:
            self.__scale_function = value

    def set_noise_type(self, noise_distribution=None):
        """
        Specify the noise distribution to add, i.e., f(x) + noise

        :param str noise_distribution:
            Noise distribution. It can be 'gaussian' or 'uniform'. The default is None.
        """
        if noise_distribution:
            self.__noise_type = noise_distribution

    def set_noise_level(self, value=None):
        """
        Specify the noise level, i.e., f(x) + value * noise

        :param float value:
            Noise level. The default is None.
        """
        if value:
            self.__noise_level = value

    def get_optimal_fitness(self):
        """
        Return the theoretical global optimum value.

        **Note:** Not all the functions have recognised theoretical optima.

        :return: float
        """
        return self.optimal_fitness

    def get_optimal_solution(self):
        """
        Return the theoretical solution.

        **Note:** Not all the functions have recognised theoretical optima.

        :return: numpy.array
        """
        return self.optimal_solution

    def get_search_range(self):
        """
        Return the problem domain given by the lower and upper boundaries, both are 1-by-variable_num arrays.

        :returns: numpy.array, numpy.array
        """
        return self.min_search_range, self.max_search_range

    def set_search_range(self, min_search_range, max_search_range):
        """
        Define the problem domain given by the lower and upper boundaries. They could be 1-by-variable_num arrays or
        floats.

        :param min_search_range:
            Lower boundary of the problem domain. It can be a numpy.array or a float.
        :param max_search_range:
            Upper boundary of the problem domain. It can be a numpy.array or a float.

        :return: None.
        """
        if isinstance(min_search_range, (float, int)) and isinstance(max_search_range, (float, int)):
            self.min_search_range = np.array([min_search_range] * self.variable_num)
            self.max_search_range = np.array([max_search_range] * self.variable_num)
        else:
            if (len(min_search_range) == self.variable_num) and (len(max_search_range) == self.variable_num):
                self.min_search_range = min_search_range
                self.max_search_range = max_search_range
            else:
                print('Invalid range!')

    def get_func_val(self, variables, *args):
        """
        Evaluate the problem function without considering additions like noise, offset, etc.

        :param numpy.array variables:
            The position where the problem function is going to be evaluated.

        :param args:
            Additional arguments that some problem functions could consider.

        :return: float
        """
        return -1

    def get_function_value(self, variables, *args):
        """
        Evaluate the problem function considering additions like noise, offset, etc. This method calls ``get_func_val``.

        :param numpy.array variables:
            The position where the problem function is going to be evaluated.

        :param args:
            Additional arguments that some problem functions could consider.

        :return: float
        """
        if isinstance(variables, list):
            variables = np.array(variables)

        # Apply modifications to the position
        variables = self.__scale_domain * variables + self.__offset_domain

        # Check which kind of noise to use
        if self.__noise_type in ['gaussian', 'normal', 'gauss']:
            noise_value = np.random.randn()
        else:
            noise_value = np.random.rand()

        # Call ``get_func_val``with the modifications
        return self.__scale_function * self.get_func_val(variables, *args) + self.__noise_level * noise_value + \
               self.__offset_function

    def plot(self, samples=55, resolution=100):
        """
        Plot the current problem in 2D.

        :param int samples: Optional.
            Number of samples per dimension. The default is 55.

        :param int resolution: Optional.
            Resolution in dpi according to matplotlib.pyplot.figure(). The default is 100.

        :return: matplotlib.pyplot
        """
        # Generate the samples for each dimension.
        x = np.linspace(self.min_search_range[0], self.max_search_range[0], samples)
        y = np.linspace(self.min_search_range[1], self.max_search_range[1], samples)

        # Create the grid matrices
        matrix_x, matrix_y = np.meshgrid(x, y)

        # Evaluate each node of the grid into the problem function
        matrix_z = []
        for xy_list in zip(matrix_x, matrix_y):
            z = []
            for xy_input in zip(xy_list[0], xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_function_value(np.array(tmp)))
            matrix_z.append(z)
        matrix_z = np.array(matrix_z)

        # Initialise the figure
        fig = plt.figure(figsize=[4, 3], dpi=resolution, facecolor='w')
        ls = LightSource(azdeg=90, altdeg=45)
        rgb = ls.shade(matrix_z, plt.cm.jet)

        # Plot data
        ax = fig.gca(projection='3d', proj_type='ortho')
        ax.plot_surface(matrix_x, matrix_y, matrix_z, rstride=1, cstride=1, linewidth=0.5,
                        antialiased=False, facecolors=rgb)  #

        # Adjust the labels
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x, y)$')
        ax.set_zlabel('$f(x, y)$')
        ax.set_title(self.func_name)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Set last adjustments
        self.plot_object = plt.gcf()
        plt.grid(linewidth=1.0)
        plt.gcf().subplots_adjust(left=0.05, right=0.85)
        plt.show()

        # Return the object for further modifications or for saving
        return self.plot_object

    # TODO: Improve function to generate better images
    def save_fig(self, samples=100, resolution=333, ext='png'):
        """
        Save the 2D representation of the problem function. There is no requirement to plot it before.

        :param int samples: Optional.
            Number of samples per dimension. The default is 100.
        :param int resolution: Optional.
            Resolution in dpi according to matplotlib.pyplot.figure(). The default is 333.
        :param str ext: Optional.
            Extension of the image file. The default is 'png'

        :return: None.
        """
        # Verify if the path exists
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        # Verify if the figure was previously plotted. If not, then do it
        # if self.plot_object is None:
        self.plot(samples, resolution)

        # Save it
        plt.tight_layout()
        self.plot_object.savefig(self.save_dir + self.func_name + '.' + ext)
        plt.show()

    def get_formatted_problem(self, is_constrained=True):
        """
        Return the problem in a simple format to be used in a solving procedure. This format contains the ``function``
        in lambda form, the ``boundaries`` as a tuple with the lower and upper boundaries, and the ``is_constrained``
        flag.

        :param bool is_constrained: Optional.
            Flag indicating if the problem domain has hard boundaries.

        :return: dict.
        """
        # TODO: Include additional parameters to build the formatted problem, e.g., length scale feature.
        return dict(function=lambda x: self.get_function_value(x),
                    boundaries=(self.min_search_range, self.max_search_range),
                    is_constrained=is_constrained)


# %% SPECIFIC PROBLEM FUNCTIONS
# 1 - Class Ackley 1 function
class P1(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([2.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True,
                         'is_constrained':True}

    def get_func_val(self, variables, *args):
        L0=12
        T0=1
        alfa=0.4
        G1=co.tf([1,1],[1,3,2,0])
        #G1=co.tf([4],[1,0.5,0])
        t=np.linspace(0,15,1000)
        GP=variables[0]*co.tf([1],1)
        GI=variables[1]*co.tf([1],[1,0])
        GD=variables[2]*co.tf([1,0],[1])
        GPID=GP+GI+GD
        Gc1=co.feedback(GPID*G1,1)
        t,yout=co.step_response(Gc1,t) 
        L=(yout.max()/yout[-1]-1)*100
        Hi = stepinfo(Gc1)
        Ts=Hi['SettlingTime']
        #L=Hi['Overshoot']
        #MP=L
        if np.isnan(Ts):
            Ts=10  
        if np.isnan(L):
            L=10  
        fcost= alfa*np.abs(L-L0)/L0 + (1-alfa)*np.abs(Ts-T0)/T0 
        return fcost
    
    
class P2(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([50.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
         #self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True,
                         'is_constrained':True}

    def get_func_val(self, variables, *args):
        L0=16
        T0=0.005
        alfa=0.3
        ref=1
        E = 200;
        L = 400e-6;
        r = 0.1;
        C = 25e-6;
        RL= 100;
        Fo= 10e3   # Frecuencia de Operacion
        To= 1.0/Fo # Periodo de Operacion
        N = 300   # Numero de ciclos;
        DW = 0.50;
        Ns = 200; # Muestro
        t=np.arange(0,N*To,N*To/(N*Ns-1))
        #AS= lambda uk: np.array([[-r/L, -uk/L],[uk/C,-1/(RL*C)]])
        AS= lambda uk: np.array([[-r/L, -(1-DW)/L],[(1-DW)/C,-1/(RL*C)]])
        A1=AS(1)
        Bs = np.array([[E/L],[0]])
        Cs = np.array([[1,0],[0,1]])
        Ds =0;
        SYS1 = ss(A1, Bs, Cs, Ds) 
        G1=co.tf(SYS1)
        GP=variables[0]*co.tf([1],1)
        GI=variables[1]*co.tf([1],[1,0])
        GD=variables[2]*co.tf([1,0],[1])
        GPID=GP+GI+GD
        Gc1=co.feedback(GPID*G1[1,0],1)
        #t=np.linspace(0,1,10000)
        t45,yout=co.step_response(Gc1)     
        #t,yout=co.step_response(Gc1,t) 
        #L=(yout.max()/yout[-1]-1)*100
        Hi = stepinfo(Gc1)
        Ts=Hi['SettlingTime']
        L=Hi['Overshoot']
        Er=Hi['SteadyStateValue']
        #MP=L
        if np.isnan(Ts):
            Ts=100  
        if np.isnan(L):
            L=100  
        fcost= 1.25*np.abs(L-L0)/L0 + 0.5*np.abs(Ts-T0)/T0 + 0.5*np.abs(Er-ref)/ref
        return fcost
# trabajar con polos dominantes
  
class P3(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([20.] * self.variable_num)
        self.min_search_range = np.array([0.001] * self.variable_num)
        self.optimal_solution = np.array([0.001] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P3'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        modelname='simulacion2.slx'
        fcost=eng.simulacionplanta(float(variables[0]),float(variables[1]),float(variables[2]),'simulacion2.slx') 
        return fcost
# trabajar con polos dominantes
    
class P4(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([ 0.1,10, 0.000000000001])
        self.min_search_range = np.array([4,60,0.0007])
        self.optimal_solution = np.array([0.001] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        modelname='ClosedLoopBoostConverter.slx'  
        fcost=eng.simulacionplanta(float(variables[0]),float(variables[1]),float(variables[2]),modelname) 
        #print(fcost)
        #print(variables)
        return fcost

    
class P5(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([ 30,30, 30])
        self.min_search_range = np.array([0,0,0.0])
        self.optimal_solution = np.array([0.001] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P5'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        modelname='masaresortereal.slx'  
        fcost=eng.simulacionplantamasaresorte(float(variables[0]),float(variables[1]),float(variables[2]),modelname)
        #print(fcost)
        return fcost
    
    
    
class P6(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([ 30,30,30])
        self.min_search_range = np.array([-30,-30,-30])
        #self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P6'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        #modelname='masaresortereal.slx'  
        fcost=eng.regresionHH(float(variables[0]),float(variables[1]),float(variables[2]))
        #print(fcost)
        return fcost    
    
class P7(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        #self.max_search_range = np.array([ 30,30,30])
        #self.min_search_range = np.array([-30,-30,-30])
        #self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P7'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
     
        fcost=eng.redneuronal(W11,W12,W13,W14,W15,W16,W17,W18,W19,W110,W111,W112,W21,W22,W23,W24,W25,W26,b11,b12,b13,b14,b15,b16,b2)
        #print(fcost)
        return fcost        
    
class P8(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([3.] * self.variable_num)
        self.min_search_range = np.array([-3.] * self.variable_num)
        #self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P8'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        #modelname='masaresortereal.slx'  
        v=[]
        for i in range(len(variables)):
            v.append(float(variables[i]))
        fcost=eng.red_mat(v)
        #print(fcost)
        return fcost    
    
class P9(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([3.] * self.variable_num)
        self.min_search_range = np.array([-3.] * self.variable_num)
        #self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P9'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        data = load_iris()
        X = data.data
        y = data.target
        n_inputs = 4
        n_hidden = 18
        n_classes = 3
        num_samples = 150
        a=n_inputs*n_hidden
        b=a+n_hidden
        c=b+n_classes*n_hidden
        d=c+n_classes
        W1 = variables[0:a].reshape((n_inputs,n_hidden))
        b1 = variables[a:b].reshape((n_hidden,))
        W2 = variables[b:c].reshape((n_hidden,n_classes))
        b2 = variables[c:d].reshape((n_classes,))
        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)     # Activation in Layer 1
        logits = a1.dot(W2) + b2 # Pre-activation in Layer 2
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Compute for the negative log likelihood
        corect_logprobs = -np.log(probs[range(num_samples), y]) 
        loss = np.sum(corect_logprobs) / num_samples
        y_pred = np.argmax(logits, axis=1)
        acu=100*(y_pred == y).mean()+0.0000001
        fcost=loss+5*(100/acu -1)
        return fcost
    
    
    
class P10(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([30.] * self.variable_num)
        self.min_search_range = np.array([0.0001] * self.variable_num)
        self.func_name = 'P10'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True,
                         'is_constrained':True}

    def get_func_val(self, variables, *args):
        L0=12
        T0=12
        alfa=0.4
        k=1.2 #resorte
        b=0.6 #amortiguador
        m=2
        G1=co.tf([1],[m,b,k])
        t=np.linspace(0,40,1000)
        GP=variables[0]*co.tf([1],1)
        GI=variables[1]*co.tf([1],[1,0])
        GD=variables[2]*co.tf([1,0],[1])
        GPID=GP+GI+GD
        Gc1=co.feedback(GPID*G1,1)
        t,yout=co.step_response(Gc1,t) 
        #plt.plot(t,yout)
        #plt.grid(True)
        #L=(yout.max()/yout[-1]-1)*100
        Hi = stepinfo(t,yout)
        L=Hi['Overshoot']
        Ts=Hi['SettlingTime']
        if np.isnan(Ts):
            Ts=100  
        if np.isnan(L):
            L=100  
        fcost= alfa*np.abs(L-L0)/L0 + (1-alfa)*np.abs(Ts-T0)/T0 
        return fcost
    
class P11(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P11'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        fcost=eng.HHbuck(float(variables[0]),float(variables[1]),float(variables[2]))
        return fcost  
    
class P12(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.3,15.0])
        self.min_search_range = np.array([0.9,1.0])
        self.global_optimum_solution = 0.
        self.func_name = 'P12'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        fcost=eng.fracHH(float(variables[0]),float(variables[1]))
        return fcost    
                
        
class P13(BasicProblem):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([ 14.0,1.812168e+03, 0.1])
        self.min_search_range = np.array([  6.0,1000.0, 4.415e-02])
        #1.23473940e+01, 1.71216800e+03, 9.41171175e-02
         #self.optimal_solution = np.array([0.001] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'P13'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        fcost=eng.resortelectrico(float(variables[0]),float(variables[1]),float(variables[2]))
        #print(fcost)
        return fcost
         
        
# %% TOOLS TO HANDLE THE PROBLEMS
def list_functions(rnp=True, fts=None, wrd='1'):
    """
    This function lists all available functions in screen. It could be formatted for copy and paste in a latex document.

    :param bool rnp: Optional.
        Flag (return-not-print). If True, the function delivers a list but not print, otherwise, print but not return.
        An example of the list returned when rnp = True is:
            [[function1_weight, function1_id, function1_name, function1_features],
             [function2_weight, function2_id, function2_name, function2_features],
             ...
             [functionN_weight, functionN_id, functionN_name, functionN_features]]

    Weights are determined with ``function.get_features("string", wrd=wrd, fts=fts)``

    :param list fts: Optional.
        Features to export/print. Possible options: 'Continuous', 'Differentiable','Separable', 'Scalable', 'Unimodal',
        'Convex'. Default: ['Differentiable','Separable', 'Unimodal']

    :return: list or none.
    """
    # Set the default value
    if fts is None:
        fts = ['Differentiable', 'Separable', 'Unimodal']

    # Initialise the variables
    feature_strings = list()
    functions_features = dict()

    # For all the functions
    for ii in range(len(__all__)):
        # Get the name and initialise its object in two dimensions
        function_name = __all__[ii]
        funct = eval("{}(2)".format(function_name))

        # Get the features and weights
        feature_str = funct.get_features(fts=fts)
        weight = funct.get_features("string", wrd=wrd, fts=fts)
        functions_features[function_name] = dict(**funct.features, Code=weight)

        # Build the list
        feature_strings.append([weight, ii + 1, funct.func_name, feature_str])

    if not rnp:
        # Print first line
        print("Id. & Function Name & " + ' & '.join(fts) + " \\\\")
        for x in feature_strings:
            print("{} & {} & {} \\\\".format(*x[1:]))
    else:
        # Return the list
        return functions_features


def for_all(property, dimension=2):
    """
    Read a determined property or attribute for all the problems and return a list.

    :param str property:
        Property to read. Please, check the attributes from a given problem object.

    :param int dimension: Optional
        Dimension to initialise all the problems.

    :return: list
    """
    if property == 'features':
        return list_functions(rnp=True, fts=None)
    else:
        info = dict()
        # Read all functions and request their optimum data
        for ii in range(len(__all__)):
            function_name = __all__[ii]
            info[function_name] = eval('{}({}).{}'.format(function_name, dimension, property))

        return info
