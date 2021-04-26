# Python Implementation for a Self-Organizing Map
# Chris Avery: April 26, 2021

import numpy as np
import matplotlib.pyplot as plt

class SOM_Model:
    def __init__(self, size, in_dim, sigma, lr, scale):
        self.size = size;
        self.sigma = sigma;
        self.lr = lr;
        if scale == "minmax":
            self.map = np.random.rand(size, size, in_dim);
        elif scale == "standard":
            self.map = 2*(np.random.rand(size, size, in_dim)-.5);
    
    def forward(self, sample):
        ans = np.zeros(shape=(self.size, self.size));
        for i in range(self.size):
            for j in range(self.size):
                ans[i,j] = self.eval_node(self.map[i,j,], sample);       
        return ans;
    
    def update_weights(self, sample):
        clust = self.forward(sample);
        bmu = self.find_bmu(clust);
        nbh = self.find_bmu_neighborhood(bmu);
        
        for i in range(self.size):
            for j in range(self.size):
                delta = nbh[i,j]*self.lr*(sample-self.map[i,j,]);
                self.map[i,j,] = self.map[i,j,] + delta;
    
    def eval_node(self, node, sample):
        node = np.reshape(node, -1);
        sample = np.reshape(sample, -1);
        d = self.distance(node, sample);
        return d;
    
    def distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2));
    
    def find_bmu(self, matrix2D):
        index = np.where(matrix2D==np.amin(matrix2D));
        return np.reshape(np.asarray(index), -1);
    
    def find_bmu_neighborhood(self, bmu):
        neighbors = np.zeros(shape=(self.size, self.size));
        for i in range(self.size):
            for j in range(self.size):
                dist = np.sqrt(np.sum(([i, j] - bmu)**2));
                if dist < self.sigma:
                    neighbors[i, j] = 1*np.exp(-.5*dist);
        return neighbors;
    
    def map2latent(self, samples):
        countmap = np.zeros(shape=(self.size, self.size))
        for i in range(samples.shape[0]):
            out = self.forward(samples[i]);
            bmu = self.find_bmu(out);
            countmap[bmu[0], bmu[1]] += 1;
        return countmap;
    
    def threeclass_plot(self, samp_tuple):
        if len(samp_tuple) == 3:
            countmaps = list()
            for i in samp_tuple:
                tmpmap = self.map2latent(i);
                countmaps.append(tmpmap/np.amax(tmpmap));
            
            color = ('Reds', 'Greens', 'Blues');
            extent = 0, self.size, 0, self.size
            
            plt.figure; 
            plt.imshow(countmaps[0], alpha=.7, cmap=color[0], 
                       interpolation='bilinear', extent=extent);
            plt.imshow(countmaps[1], alpha=.5, cmap=color[1], 
                       interpolation='bilinear', extent=extent);
            plt.imshow(countmaps[2], alpha=.3, cmap=color[2], 
                       interpolation='bilinear', extent=extent);
            plt.show()
            
    
    
