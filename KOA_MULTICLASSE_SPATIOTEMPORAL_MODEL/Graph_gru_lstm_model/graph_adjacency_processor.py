import tensorflow as tf
import numpy as np
from IPython.core.debugger import set_trace

class Graph():
    def __init__(self, num_node):
        self.num_node = num_node
        self.AD, self.AD2, self.bias_mat_1, self.bias_mat_2 = self.normalize_adjacency()
        
    def normalize_adjacency(self):
        self_link = [(i, i) for i in range(self.num_node)]
        # neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        #                               (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        #                               (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        #                               (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        #                               (22, 23), (23, 8), (24, 25), (25, 12)]

        neighbor_1base = [(1, 2), (2, 3),(3, 4),(4, 8),(1, 5),(5, 6),(6, 7), (7, 9),(10, 11),
                 (12, 13),(12, 14),(14, 16),(16, 22),(16, 18),(16, 20),(18, 20),(13, 15),(15, 17),(17, 23),(17, 19),(17, 21),(21, 19)
                  ,(12, 24),(24, 25),(24, 26),(26, 28),(28, 30),(28, 32),(30, 32),
                  (13, 25),(25, 27),(27, 29),(29, 31),(29, 33),(31, 33)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]

        # neighbor_1base = [(12, 13),(12, 14),(14, 16),(16, 22),(16, 18),(16, 20),(18, 20),(13, 15),(15, 17),(17, 23),(17, 19),(17, 21),(21, 19)
        #           ,(12, 24),(24, 25),(24, 26),(26, 28),(28, 30),(28, 32),(30, 32),
        #           (13, 25),(25, 27),(27, 29),(29, 31),(29, 33),(31, 33)]
        # neighbor_link = [(i - 12, j - 12) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link    
        A = np.zeros((self.num_node, self.num_node)) # adjacency matrix
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        
        A2 = np.zeros((self.num_node, self.num_node)) # second order adjacency matrix
        for root in range(A.shape[1]):
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root,neighbour_of_neigbour] = 1                 
        #AD = self.normalize(A)
        #AD2 = self.normalize(A2)
        bias_mat_1 = np.zeros(A.shape)
        bias_mat_2 = np.zeros(A2.shape)
        bias_mat_1 = np.where(A!=0, bias_mat_1, -1e9)
        bias_mat_2 = np.where(A2!=0, A2, -1e9)
        AD = A.astype('float32')
        AD2 = A2.astype('float32')
        bias_mat_1 = bias_mat_1.astype('float32')
        bias_mat_2 = bias_mat_2.astype('float32')
        AD = tf.convert_to_tensor(AD)
        AD2= tf.convert_to_tensor(AD2)
        bias_mat_1 = tf.convert_to_tensor(bias_mat_1)
        bias_mat_2 = tf.convert_to_tensor(bias_mat_2)
        return AD, AD2, bias_mat_1, bias_mat_2
        
    def normalize(self, adjacency):
        rowsum = np.array(adjacency.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = np.diag(r_inv)
        normalize_adj = r_mat_inv.dot(adjacency)
        normalize_adj = normalize_adj.astype('float32')
        normalize_adj = tf.convert_to_tensor(normalize_adj)   
        return normalize_adj