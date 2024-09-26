import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from data_driven_legged_locomotion.maps.h1_walk import Map

class GlobalPlanner:
    def __init__(self, map: Map, start_pos: np.ndarray, goal_pos: np.ndarray):
        if map is None:
            self.map = Map()
        else:
            self.map = map
        self.start_pos = start_pos
        self.goal_pos = goal_pos
    
    def _cost_func(self, pos: np.ndarray):
        return self.map.cost(pos) + np.linalg.norm(pos - self.goal_pos)
    
    def plot_cost(self):
        path = self.get_path()
        X, Y = np.meshgrid(np.linspace(-1, 11, 100), np.linspace(-1, 11, 100))
        z = np.array([self._cost_func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = z.reshape(X.shape)
        grad = np.gradient(Z)
        fig_3d = plt.figure()
        ax = fig_3d.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z)
        fig_grad = plt.figure()
        ax = fig_grad.add_subplot()
        ax.quiver(X, Y, -grad[1], -grad[0], angles='xy')
        ax.plot(path[:,0], path[:,1], 'r')
        fig_heatmap = plt.figure()
        ax = fig_heatmap.add_subplot()
        ax.imshow(Z, extent=[-1, 11, -1, 11], origin='lower')
        ax.plot(path[:,0], path[:,1], 'r')
        plt.show()
        return fig_3d, fig_grad, fig_heatmap
    
    def get_path(self, n_bins = 100):
        """Gets a path from starting_point to goal_point using A*"""
        space_lengths = self.map.extreme_points[1] - self.map.extreme_points[0]
        step_sizes = space_lengths / n_bins
        starting_node = tuple(np.floor(self.start_pos / step_sizes).astype(int))
        goal_node = tuple(np.floor(self.goal_pos / step_sizes).astype(int))
        G = nx.generators.grid_2d_graph(n_bins+1, n_bins+1)
        for source,dest,data_dict in G.edges(data=True):
            data_dict['weight'] = self._cost_func(np.array([dest[0]*step_sizes[0], dest[1]*step_sizes[1]]))
        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            x1 *= step_sizes[0]
            y1 *= step_sizes[1]
            x2 *= step_sizes[0]
            y2 *= step_sizes[1]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        path = nx.astar_path(G, starting_node, goal_node, heuristic=dist, weight="weight")
        path_arr = np.array(path) * step_sizes
        return path_arr