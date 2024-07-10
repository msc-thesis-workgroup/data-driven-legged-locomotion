import numpy as np
import math

from abc import ABC, abstractmethod

N_SAMPLES = 20

from scipy.integrate import quad

def kl_divergence_continuous(p, q, lower_bound: np.array, upper_bound:np.array):
    """Calculates the Kullback-Leibler divergence between two continuous distributions."""
    kl_div = 0
    for i in range(len(p)):
        kl_div += quad(lambda x: p[i](x) * math.log(p[i](x) / q[i](x)), lower_bound[i], upper_bound[i])
    return kl_div
    

def kl_divergence_monte_carlo_continuous(p, q,num_samples):
    samples = p.sample(num_samples)


class CostFunction(ABC):
    """Abstract class for cost functions."""
    @abstractmethod
    def cost_function(x):
        pass
class Sources(ABC):
    """Abstract class to define the policy sources."""
    @abstractmethod
    def return_sources(state: np.array):
        """Return the sources for the given state."""
        pass

    @abstractmethod
    def get_relative_state_space_dimension():
        pass

    @abstractmethod
    def sample_points(self,i:int = 0,num_points: int = 1):
        """Samples points from the sources."""
        pass

class TargetBehavior(ABC):
    """Abstract class to define the target behavior."""
    # TODO(all): Refine the goal of this class
    @abstractmethod
    def target_behavior():
        pass        


class CrowdSourcing:

    def __init__(self, cost_function: CostFunction, sources: Sources, target_behavior: TargetBehavior):
        self.cost_function = cost_function
        self.sources = sources
        self.target_behavior = target_behavior


    def cost_function_time_varying(self,x, k: int):
        """ Selects the cost function based on the time step k."""
        
        # for now we are using the same cost function for all time steps
        return self.cost_function.cost_function(x)
    
    # def execute(self, xStart):
    #     """Executes the crowd sourcing algorithm."""
        
    #     #timeHor = list(range(tHor))
    #     #timeHor.reverse()
    #     S = self.sources.return_sources() # As the notation in the paper

    #     xDim = self.sources.get_relative_state_space_dimension() # TODO(): xdim dovrebbe essere l'observation space
    #     # Initialize the relative state space

    #     rHat = [0]*xDim #Reward modifier, same dimension as the relative state space
    #     weights = np.zeros((xDim, len(S))) # := the vector 'a' in the paper. In reality it is a matrix with xDim rows and S columns

    #     #if tHor == 0: # If the time horizon is 0 timeHor would be an empty list.
    #     #    timeHor = [0]
        
    #     #for k in timeHor:
            
    #         # If I am correct, now we have to loop over all the sources and 
    #         # sample 20 points for each source Si and compute the next state (x_k) applying the monte carlo method
            

    #         # Quesito CRUCIALE: l'algoritmo prevede di partire da N (timeHor-1) e poi scendere fino a 1 (0), ma non è chiaro come si possa fare
    #         # dato che ho solo lo stato attuale(x_k-1). In altre parole come posso ottenere lo stato x_{timeHor -1} se non so come evolve allo stato successivo?
    #         # Devo prendere ogni possibile stato? Ma lo stato è continuo.
    #         # Posso partire da x_k-1 e fare un passo avanti prendendo x_k dato x_k-1 e poi posso continuare fino a x_{timeHor-1}?
    #         # Ho appena controllato EMILAND GARRABE fa così. Anche secondo me è l'unica soluzione.


    #     for i in range(len(S)):
             
    #         x_k = xStart

    #         r_hat[x_k] = -min(weights[x_k])



    #         #act_state = actualState(xStart,x)
    #         rHat[x] = -min(weights[x])
    #         act_state = actualState(xStart,x)
    #         r = np.add(return_reward(act_state), rHat)
    #         sources = return_sources(act_state)
    #         target = return_target(act_state)
    #         #fare un vettore che va da 0 a S e metterci il vettore a e poi estrarre l'indice per cui è minimo
    #         for i in range(S):
    #             weights[x,i] = DKL(sources[i], target) +- np.dot(sources[i], r)
    #     if k==0:
    #         j = np.argmin(weights[0])
    #         return(return_sources(xStart)[j])
    
    # def receding_horizon_DM(self, state, tHor, stateSpace, r):
    #     xInd = np.where(stateSpace == state)[0][0] #Index of the state in the reduced state space

    #     num_sources = self.sources.shape[0]
    #     full_state_space_dim = self.sources.shape[1] 
        
    #     dim = len(stateSpace)
    #     rHat = np.array([0]*full_state_space_dim) #rHat as in the algorithm 
    #     weights = np.zeros((num_sources, dim)) #Decision-making weights

    #     sources = np.zeros((num_sources, dim, full_state_space_dim)) #The sources and target we use correspond to the reduced state space
    #     target = np.zeros((dim,full_state_space_dim))
        
    #     self.PMF_target = behaviors[self.nTarget]
    #     for i in range(dim):
    #         target[i] = self.PMF_target[stateSpace[i]]


    #     for i in range(num_sources):
    #         for j in range(dim):
    #             sources[i,j] = behaviors[i,stateSpace[j]]
    #     if tHor == 0: #Safety as python doesn't like range(0)
    #         tHor = 1
    
    #     timeHor = list(range(tHor))
    #     timeHor.reverse()
    
    #     for t in timeHor:
    #         rBar = r + rHat #Adapt reward
    #         for i in range(num_sources):
    #             for j in range(dim):
    #                 weights[i,j] = self.__DKL(sources[i,j], target[j]) - np.dot(sources[i,j], rBar) #calculate weights
    #         for i in range(dim):
    #             rHat[stateSpace[i]] = -min(weights[:,i]) #Calculate rHat
    #     indMin = np.argmin(weights[:,xInd])
    #     pf = sources[indMin, xInd] #Pick pf
    #     return(np.random.choice(range(full_state_space_dim), p = pf)) #Sample pf and return resulting state

    def __DKL(self, l1, l2):
        """Calculates the Kullback-Leibler divergence between two arrays."""
        x = 0
        for i in range(len(l1)):
            if l1[i] != 0 and l2[i] != 0:
                x = x + l1[i] * math.log(l1[i] / l2[i])
            if l2[i] == 0 and l1[i] != 0:
                return math.inf
        return x

    def execute_greedy(self, x):
        """Executes the greedy algorithm."""
        sources = self.sources.return_sources(x)
        num_sources = len(sources)
        obs_space = len(x)
        weights = np.zeros((num_sources, obs_space))
        target = self.target_behavior.target_behavior()

        # check if the target is compatible with the observation space
        
        lower_bound = sources.lower_bound
        upper_bound = sources.upper_bound
        for i in range(num_sources):
            
            weights[i] = kl_divergence_continuous(sources[i], target, lower_bound, upper_bound)
            # calculate the expected cost given the state x and the source i

            # To calculate the expected reward we need to calculate the expected cost through the monte carlo method
            
            samples = self.sources.sample_points(x, i, N_SAMPLES)
            # calculate the expected cost
            expected_cost = 0
            for s in samples:
                expected_cost += self.cost_function.cost_function(s)
            expected_cost /= N_SAMPLES

            weights[i] -= expected_cost

        # select the source with the minimum weight
        min_index = np.argmin(weights)
        return min_index
            
