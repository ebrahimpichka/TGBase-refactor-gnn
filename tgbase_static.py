import argparse
import datetime
import sys
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
from dataclasses import dataclass
import multiprocessing as mp

from utils import *

@dataclass
class EdgeAttributes:
    timestamp: list
    weight: list

@dataclass
class EdgeStats:
    avg: float
    min: float
    max: float
    sum: float
    std: float
    ent: float

@dataclass
class NodeFeatures:
    node: int
    degree: int
    in_degree: int
    out_degree: int
    w_in: EdgeStats
    w_out: EdgeStats
    w_all: EdgeStats
    t_in: EdgeStats
    t_out: EdgeStats
    t_all: EdgeStats
    neighbor_degree: EdgeStats
    neighbor_in_degree: EdgeStats
    neighbor_out_degree: EdgeStats

class StaticTGBase:
    def __init__(self, network_name, edgelist_path):
        self.edgelist_path = edgelist_path
        self.network_name = network_name

    def get_edge_attributes(self, edgelist_df, node, e_attr_opt, dir_opt):
        """
        Get the list of edge attributes of a given node

        :param edgelist_df: the edgelist dataframe
        :param node: the given node
        :param e_attr_opt: the edge attribute option
        :param dir_opt: the direction option
        :return: the list of edge attributes
        """

        if dir_opt == 'in':
            e_feat_list = edgelist_df.loc[edgelist_df['target'] == node, e_attr_opt].tolist()
        elif dir_opt == 'out':
            e_feat_list = edgelist_df.loc[edgelist_df['source'] == node, e_attr_opt].tolist()
        elif dir_opt == 'all':
            e_feat_list = edgelist_df.loc[((edgelist_df['source'] == node) | (edgelist_df['target'] == node)), e_attr_opt].tolist()
        else:
            raise ValueError("Undefined option!")

        if e_attr_opt == 'timestamp':
            timestamp_list = [datetime.datetime.fromtimestamp(t) for t in e_feat_list]
            timestamp_list.sort()
            e_feat_list = [((timestamp_list[i + 1] - timestamp_list[i]).total_seconds() / 60) for i in range(len(timestamp_list) - 1)]

        return e_feat_list

    def generate_edge_attr_feats(self, edgelist_df, node, e_attr_opt):
        """
        Generate edge attribute features for a given node

        :param edgelist_df: the edgelist dataframe
        :param node: the given node
        :param e_attr_opt: the edge attribute option
        :return: the edge attribute features
        """

        in_edge_attr_list = self.get_edge_attributes(edgelist_df, node, e_attr_opt, 'in')
        in_edge_stats = EdgeStats(*agg_stats_of_list(in_edge_attr_list))

        out_edge_attr_list = self.get_edge_attributes(edgelist_df, node, e_attr_opt, 'out')
        out_edge_stats = EdgeStats(*agg_stats_of_list(out_edge_attr_list))

        all_edge_attr_list = self.get_edge_attributes(edgelist_df, node, e_attr_opt, 'all')
        all_edge_stats = EdgeStats(*agg_stats_of_list(all_edge_attr_list))

        return in_edge_stats, out_edge_stats, all_edge_stats

    def get_neighborhood_feat(self, G, node):
        """
        Get the neighborhood features of a given node
        
        :param G: the graph
        :param node: the given node
        :return: the neighborhood features
        """

        neighbors = list(G.neighbors(node))

        degree_nodeView = G.degree(neighbors)
        degree_list = list(dict(degree_nodeView).values())
        degree_feats = EdgeStats(*agg_stats_of_list(degree_list))

        in_degree_nodeView = G.in_degree(neighbors)
        in_degree_list = list(dict(in_degree_nodeView).values())
        in_degree_feats = EdgeStats(*agg_stats_of_list(in_degree_list))

        out_degree_nodeView = G.out_degree(neighbors)
        out_degree_list = list(dict(out_degree_nodeView).values())
        out_degree_feats = EdgeStats(*agg_stats_of_list(out_degree_list))

        return degree_feats, in_degree_feats, out_degree_feats

    def generate_node_feats(self, G, edgelist_df, node):
        """
        Generate node features for a given node

        :param G: the graph
        :param edgelist_df: the edgelist dataframe
        :param node: the given node
        :return: the node features
        """

        stats_time_in, stats_time_out, stats_time_all = self.generate_edge_attr_feats(edgelist_df, node, 'timestamp')
        stats_w_in, stats_w_out, stats_w_all = self.generate_edge_attr_feats(edgelist_df, node, 'weight')
        nei_degree_feats, nei_in_degree_feats, nei_out_degree_feats = self.get_neighborhood_feat(G, node)

        node_feats = NodeFeatures(
            node=node,
            degree=G.degree(node),
            in_degree=G.in_degree(node),
            out_degree=G.out_degree(node),
            w_in=stats_w_in,
            w_out=stats_w_out,
            w_all=stats_w_all,
            t_in=stats_time_in,
            t_out=stats_time_out,
            t_all=stats_time_all,
            neighbor_degree=nei_degree_feats,
            neighbor_in_degree=nei_in_degree_feats,
            neighbor_out_degree=nei_out_degree_feats
        )

        return node_feats

    def generate_all_nodes_feat(self, G, edgelist_df):
        """
        Generate node features for all nodes
            
        :param G: the graph
        :param edgelist_df: the edgelist dataframe
        :return: the node features dataframe
        """

        node_list = list(G.nodes())
        node_feats_list = []
        for node in tqdm(node_list):
            node_feats = self.generate_node_feats(G, edgelist_df, node)
            node_feats_list.append(node_feats)
        node_feats_df = pd.DataFrame([f.__dict__ for f in node_feats_list])
        return node_feats_df

    def generate_graph(self):
        edgelist_df = pd.read_csv(self.edgelist_path, header=None)
        edgelist_df.columns = ['source', 'target', 'weight', 'timestamp']
        G = nx.from_pandas_edgelist(edgelist_df, source='source', target='target', edge_attr=['weight', 'timestamp'],
                                    create_using=nx.MultiDiGraph)
        return edgelist_df, G

    def save_node_feats_df(self, node_feats_df, output_path):
        node_feats_df.to_csv(output_path, index=False)

    def encode_features(self, save_to_file=False):
        """ 
        Generate TGBase features for a static network

        :param save_to_file: whether to save the features to file
        :return: the node features dataframe
        """
        edgelist_df, G = self.generate_graph()
        node_feats_df = self.generate_all_nodes_feat(G, edgelist_df)
        node_feats_df_filename = f'./data/{self.network_name}/{self.network_name}_node_feats.csv'
        if save_to_file:
            self.save_node_feats_df(node_feats_df, node_feats_df_filename)
        return node_feats_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TGBase Features for Static Networks.')
    parser.add_argument('--network', type=str, default='', help='Network name.')

    args = parser.parse_args()
    print("Arguments:", args)

    network_name = args.network
    edgelist_path = f'./data/{network_name}/{network_name}_network.csv'
    static_tg_base = StaticTGBase(network_name, edgelist_path)
    static_tg_base.encode_features()