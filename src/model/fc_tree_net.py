import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCTreeNet(torch.nn.Module):
    def __init__(self, in_dim=300, img_dim=256, use_cuda=True):
        '''
        initialization for TreeNet model, basically a ChildSumLSTM model
        with non-linear activation embedding for different nodes in the AoG.
        Shared weigths for all LSTM cells.
        :param in_dim:      input feature dimension for word embedding (from string to vector space)
        :param img_dim:     dimension of the input image feature, should be (panel_pair_number * img_feature_dim (e.g. 512 or 256))
        '''
        super(FCTreeNet, self).__init__()
        self.in_dim = in_dim
        self.img_dim = img_dim
        self.fc = nn.Linear(self.in_dim, self.in_dim)
        self.leaf = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.middle = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.merge = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.root = nn.Linear(self.in_dim + self.img_dim, self.img_dim)

        self.relu = nn.ReLU()

    def forward(self, image_feature, node_label, indicator):
        '''
        Forward funciton for TreeNet model
        :param node_label:		input should be (batch_size * 6 * input_word_embedding_dimension), got from the embedding vector
        :param indicator:	indicating whether the input is of structure with branches (batch_size * 1)
        :param image_feature:   input dictionary for each node, primarily feature, for example (batch_size * 16 (panel_pair_number) * feature_dim (output from CNN))
        :return:
        '''
        # image_feature = image_feature.view(-1, 16, image_feature.size(2))
        node_label = self.fc(node_label.view(-1, node_label.size(-1)))
        node_label = node_label.view(-1, 6, node_label.size(-1))
        node_label = node_label.unsqueeze(1).repeat(1, image_feature.size(1), 1, 1)
        indicator = indicator.unsqueeze(1).repeat(1, image_feature.size(1), 1).view(-1, 1)

        leaf_left = node_label[:, :, 3, :].view(-1, node_label.size(-1))           # (batch_size * panel_pair_num) * input_word_embedding_dimension
        leaf_right = node_label[:, :, 5, :].view(-1, node_label.size(-1))
        inter_left = node_label[:, :, 2, :].view(-1, node_label.size(-1))
        inter_right = node_label[:, :, 4, :].view(-1, node_label.size(-1))
        merge = node_label[:, :, 1, :].view(-1, node_label.size(-1))
        root = node_label[:, :, 0, :].view(-1, node_label.size(-1))
        
        # concating image_feature and word_embeddings for leaf node inputs
        leaf_left = torch.cat((leaf_left, image_feature.view(-1, image_feature.size(-1))), dim=-1)
        leaf_right = torch.cat((leaf_right, image_feature.view(-1, image_feature.size(-1))), dim=-1)

        out_leaf_left = self.leaf(leaf_left)
        out_leaf_right = self.leaf(leaf_right)

        out_leaf_left = self.relu(out_leaf_left)
        out_leaf_right = self.relu(out_leaf_right)

        out_left = self.middle(torch.cat((inter_left, out_leaf_left), dim=-1))
        out_right = self.middle(torch.cat((inter_right, out_leaf_right), dim=-1))

        out_left = self.relu(out_left)
        out_right = self.relu(out_right)

        out_right = torch.mul(out_right, indicator)
        merge_input = torch.cat((merge, out_left + out_right), dim=-1)
        out_merge = self.merge(merge_input)

        out_merge = self.relu(out_merge)

        out_root = self.root(torch.cat((root, out_merge), dim=-1))
        out_root = self.relu(out_root)
        # size ((batch_size * panel_pair) * feature_dim)
        return out_root




# might need to refactor this as a subclass of nn.Moulde
# with the first two arguments combined and the last as a self class
# but a problem is that if the input x is a batch, then each instance in the batch
# might have different L. So, maybe, I need to change the dataset loading part.
def tree_net_forward(L: list[str], x: Tensor, vertex_nets: dict[str, nn.Module]):
    '''
    (1) Restore the tree structure according to the list L of vertices obtained by preorder traversing the tree,
    (2) populate each vertex with a neural net work, which is determined by the name and/or type of the vertex,
    i.e., the elements in L.
    (3) pass the input tensor x from leaf vertices to the root, which provides the output

    :param L: list of vectices obtained by preorder traversal. For example,
    [b'Scene', b'Out_In', b'Out', b'Out_Center_Single', b'/', b'/', b'In', b'In_Distribute_Four', b'/', b'/', b'/', b'/'],
    the '/' denoting the end-of-branch and  it is the time to backtrack. Note that, generally, preorder traversal strings
    are not able to uniquely determine the tree structure, but with these '/', the strings are able to do so.
    We assume that L is a correct preorder traverse of a non-empty tree, i.e., |L| >= 2 with the minimum like ['Root', '/']

    :param x: input tensor X with shape [batch_size, feature_size] or [batch_size, channel_number, feature_size].
    The tree net processes each feature in the same way. Note that all the instance in the batch must share the
    same tree structure, i.e., L. This implies that the batches must be taken from a single spatial configuration.
    This requires some modification of dataloaders.

    :param vertex_nets: a dictionary containing to neural network to populate vertices, indexed by vertex names.
    Note that if the same neural net is referred to multiple keys or vertex names are not unique,
    it would introduce recurrent structure in the tree net.
    '''

    if '/' == L[1]: # might also need to check '/' != L[0], but we assume the input is correct
        return L[2:], vertex_nets.get(L[0])(x)
    else:
        R = L[1:]
        acc = 0
        while '/' != R[0]:
            R, y = tree_net_forward(R, x)
            acc += y
        return R, vertex_nets.get(L[0])(acc)

