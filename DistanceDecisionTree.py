
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import graphviz
from graphviz import Digraph



class Node:
    
#value is set as none for a non leaf node, for leaf node, it is a category
    def __init__(self, feature=None, threshold=None,  left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DistanceDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root= None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        #if depth of tree is greater than max depth or number of samples less than min
        if depth >= self.max_depth or n_samples < self.min_samples_split:
           leaf_value = self._most_common_label(y)
           return Node(value=leaf_value)

        n_labels = len(np.unique(y))
        #if node is already pure
        if n_labels == 1:
           leaf_value = y[0]
           return Node(value=leaf_value)
        feat_idxs = range(n_feats)
        # feat_idxs = np.random.permutation(n_feats)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs )
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _best_split(self, X, y, feat_idxs):
        min_penalty = 10000000
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                penalty = self._penalty(y, X_column, thr)

                if penalty <= min_penalty:
                    min_penalty = penalty
                    split_idx = feat_idx
                    split_threshold = thr



        return split_idx, split_threshold


    def _penalty(self, y, X_column, threshold):
        minpenalty = 10000000

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return minpenalty
        feature_string = ''.join(map(str, (X_column>threshold).astype(int)))
        labels = np.unique(y)
        for label in labels:
          target_y = np.where(y==label,1,0)
          target_string = ''.join(map(str, target_y))

          penalty = self._lz_complexity_penalty(feature_string, target_string)
          if penalty< minpenalty:
              minpenalty = penalty

        return minpenalty


    def _lz_complexity_penalty(self, sequence_gr, sequence_cmp):

            """Lempel-Ziv complexity for a binary sequence, in simple Python code. """
            sub_strings_gr = set()
            sub_strings_cmp = set()
            sub_strings_penalty = set()
            n = len(sequence_gr)
            ind = 0
            inc = 1
            #find grammar of x
            while True:
                if ind + inc > len(sequence_gr):
                    break
                sub_str = sequence_gr[ind : ind + inc]
                if sub_str in sub_strings_gr:
                    inc += 1
                else:
                    sub_strings_gr.add(sub_str)
                    ind += inc
                    inc = 1
            #use it to compress y
            ind = 0
            inc = 1

            while True:
                if ind + inc > len(sequence_cmp):
                    break
                sub_str = sequence_cmp[ind : ind + inc]
                if sub_str in sub_strings_gr and sub_str not in sub_strings_cmp:
                    sub_strings_cmp.add(sub_str)
                    ind += inc
                    inc = 1
                elif sub_str in sub_strings_cmp:
                    inc += 1
                else:
                    if sub_str not in sub_strings_penalty:
                      sub_strings_penalty.add(sub_str)
                      ind += inc
                      inc = 1
                    else:
                      inc += 1

            return len(sub_strings_gr)-len(sub_strings_cmp)+len(sub_strings_penalty)
#returns the most commonly occuring label in a node
    def _most_common_label(self, y):
      if len(y)==0:
          return -1
      unique_labels, counts = np.unique(y,return_counts=True)
      most_common_label = unique_labels[np.argmax(counts)]
      return most_common_label



    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])


    def _traverse_tree(self, x, node):
       if node is None:
        # Handle the case when the current node is None (reached a leaf node)
        return None
       if node.is_leaf_node():
        return node.value

       if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)
       else:
        return self._traverse_tree(x, node.right)
    def export_graphviz(self, node=None, tree_graph=None,list = None):
      if node is None:
          node = self.root

      if tree_graph is None:
          tree_graph = Digraph()
          tree_graph.node(name=str(id(node)), label=f"Root")
      if list is None:
          list = range(self.n_features)

      if node.is_leaf_node():
          tree_graph.node(name=str(id(node)), label=f"Leaf: {node.value}")
      else:
          tree_graph.node(name=str(id(node)), label=f"Feature {list[node.feature]} <= {node.threshold:.2f}")

          # Add left and right child nodes
          if node.left:
              tree_graph.edge(str(id(node)), str(id(node.left)), "True")
              self.export_graphviz(node.left, tree_graph,list=list)

          if node.right:
              tree_graph.edge(str(id(node)), str(id(node.right)), "False")
              self.export_graphviz(node.right, tree_graph,list = list)

      return tree_graph