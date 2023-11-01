import argparse
import sys
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List
from utils import get_data_node_classification

rnd_seed = 2021
random.seed(rnd_seed)

@dataclass
class NodeState:
    no_event: int = 0
    time_avg: float = 0.0
    time_min: float = float('inf')
    time_max: float = 0.0
    time_sum: float = 0.0
    time_last: float = 0.0
    time_std: float = 0.0
    features: List[float] = field(default_factory=list)

@dataclass
class EdgeFeature:
    avg: float = 0.0
    min_: float = float('inf')
    max_: float = 0.0
    sum_: float = 0.0
    std: float = 0.0

class DynamicTGBase:
    """ 
    Generate TGBase dynamic embeddings for a dataset
    """
    def __init__(self, network_name, val_ratio, test_ratio, use_validation=False ):
        """ 
        Initialize the class
        
        ::param network_name: name of the network
        ::param val_ratio: validation ratio
        ::param test_ratio: test ratio
        ::param use_validation: whether to use a validation set
        """
        self.network_name = network_name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.use_validation = use_validation

    def init_node_memory(self, node_list, no_edge_feats):
        init_node_states = {}
        for node in node_list:
            init_node_states[node] = NodeState()
            init_node_states[node].features = [EdgeFeature() for _ in range(no_edge_feats)]
        return init_node_states

    def update_node_state(self, current_node_state, timestamp, edge_feature):
        time_interval = timestamp - current_node_state.time_last
        current_node_state.no_event += 1
        current_node_state.time_avg = (current_node_state.time_avg * current_node_state.no_event + time_interval) / current_node_state.no_event
        current_node_state.time_min = min(current_node_state.time_min, time_interval)
        current_node_state.time_max = max(current_node_state.time_max, time_interval)
        current_node_state.time_sum += time_interval
        current_node_state.time_last = timestamp
        current_node_state.time_std = np.sqrt(((time_interval - current_node_state.time_avg) * (time_interval - current_node_state.time_avg) + (current_node_state.no_event - 1) * current_node_state.time_std ** 2) / current_node_state.no_event)

        for feat_idx, feat in enumerate(current_node_state.features):
            feat.avg = (feat.avg * (current_node_state.no_event - 1) + edge_feature[feat_idx]) / current_node_state.no_event
            feat.min_ = min(feat.min_, edge_feature[feat_idx])
            feat.max_ = max(feat.max_, edge_feature[feat_idx])
            feat.sum_ += edge_feature[feat_idx]
            feat.std = np.sqrt(((edge_feature[feat_idx] - feat.avg) * (edge_feature[feat_idx] - feat.avg) +\
                                 (current_node_state.no_event - 1) * feat.std ** 2) / current_node_state.no_event)

        return current_node_state

    def gen_dynamic_emb_for_data_split(self, data, node_memory, edge_features):
        """
        generate dynamic embeddings for a list of nodes
        """
        emb_list = []
        print("Info: Number of interactions:", len(data.sources))
        for idx, source in tqdm(enumerate(data.sources)):  # NB: Only "source" nodes
            prev_source_state = node_memory[source]  # current state features
            current_source_state = self.update_node_state(prev_source_state, data.timestamps[idx],
                                                        edge_features[data.edge_idxs[idx]])
            node_memory[source] = current_source_state
            current_source_state.node_id = source
            current_source_state.timestamp = data.timestamps[idx]
            current_source_state.label = data.labels[idx]
            emb_list.append(current_source_state)

        return node_memory, emb_list

    def append_mask_to_emb(self, emb_list, mask_triplet):
        for emb in emb_list:
            emb.train_mask = mask_triplet[0]
            emb.val_mask = mask_triplet[1]
            emb.test_mask = mask_triplet[2]
        return emb_list

    def generate_TGBase_DynEmb(self, save_to_file=True):
        """
        generate TGBase dynamic embeddings for a dataset
        """
        full_data, _, edge_features, train_data, val_data, test_data = get_data_node_classification(
            self.network_name, self.val_ratio, self.test_ratio, self.use_validation)

        node_list = full_data.unique_nodes
        print("Info: Total Number of nodes: {}".format(len(node_list)))
        no_edge_feats = len(edge_features[0])
        node_memory = self.init_node_memory(node_list, no_edge_feats)

        # train split
        print("Info: Generating embeddings for training set...")
        node_memory, emb_list_train = self.gen_dynamic_emb_for_data_split(train_data, node_memory, edge_features)
        train_embs = self.append_mask_to_emb(emb_list_train, (1, 0, 0))
        dy_emb_filename_train = f'./data/{self.network_name}_TGBase_emb_train.csv'
        node_emb_df_train = pd.DataFrame(
            [
                {k:v for k,v in vars(e).items() if k!='features'}|\
                    {f'feat_{i}_{fk}':fv for (i,feat) in enumerate(e.features) for (fk, fv) in vars(feat).items()}
                for e in train_embs
                ]
        )
        if save_to_file:
            node_emb_df_train.to_csv(dy_emb_filename_train, index=False)

        # val split
        print("Info: Generating embeddings for validation set...")
        node_memory, emb_list_val = self.gen_dynamic_emb_for_data_split(val_data, node_memory, edge_features)
        val_embs = self.append_mask_to_emb(emb_list_val, (0, 1, 0))
        dy_emb_filename_val = f'./data/{self.network_name}/{self.network_name}_TGBase_emb_val.csv'
        node_emb_df_val = pd.DataFrame(
            [
                {k:v for k,v in vars(e).items() if k!='features'}|\
                    {f'feat_{i}_{fk}':fv for (i,feat) in enumerate(e.features) for (fk, fv) in vars(feat).items()}
                for e in val_embs
                ]
        )

        if save_to_file:
            node_emb_df_val.to_csv(dy_emb_filename_val, index=False)

        # test split
        print("Info: Generating embeddings for test set...")
        node_memory, emb_list_test = self.gen_dynamic_emb_for_data_split(test_data, node_memory, edge_features)
        test_embs = self.append_mask_to_emb(emb_list_test, (0, 0, 1))
        dy_emb_filename_test = f'./data/{self.network_name}/{self.network_name}_TGBase_emb_test.csv'
        node_emb_df_test = pd.DataFrame(
            [
                {k:v for k,v in vars(e).items() if k!='features'}|\
                    {f'feat_{i}_{fk}':fv for (i,feat) in enumerate(e.features) for (fk, fv) in vars(feat).items()}
                for e in test_embs
                ]
            )
        if save_to_file:
            node_emb_df_test.to_csv(dy_emb_filename_test, index=False)

        return node_emb_df_train, node_emb_df_val, node_emb_df_test

    def encode_features(self, save_to_file=False):
        return self.generate_TGBase_DynEmb(save_to_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TGBase Features for Dynamic Networks.')
    parser.add_argument('--network', type=str, default='wikipedia', help='Network name')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio')
    parser.add_argument('--use_validation', action='store_true', help='Whether to use a validation set')

    args = parser.parse_args()
    print("Arguments:", args)

    dynamic_tg_base = DynamicTGBase(args.network, args.val_ratio, args.test_ratio, args.use_validation)
    dynamic_tg_base.encode_features()