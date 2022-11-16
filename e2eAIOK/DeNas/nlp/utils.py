import ast
import time
import random
import numpy as np
import json
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"

def generate_search_space(search_space_config):
        # build arch space
        search_space = {}
        search_space['layer_num'] = range(int(search_space_config['LAYER_NUM']['bounds']['min']), int(search_space_config['LAYER_NUM']['bounds']['max'])+1)
        search_space['head_num'] = range(int(search_space_config['HEAD_NUM']['bounds']['min']), int(search_space_config['HEAD_NUM']['bounds']['max']) + int(search_space_config['HEAD_NUM']['bounds']['step']), int(search_space_config['HEAD_NUM']['bounds']['step']))
        search_space['hidden_size'] = range(int(search_space_config['HIDDEN_SIZE']['bounds']['min']), int(search_space_config['HIDDEN_SIZE']['bounds']['max']) + int(search_space_config['HIDDEN_SIZE']['bounds']['step']), int(search_space_config['HIDDEN_SIZE']['bounds']['step']))
        search_space['ffn_size'] = range(int(search_space_config['INTERMEDIATE_SIZE']['bounds']['min']), int(search_space_config['INTERMEDIATE_SIZE']['bounds']['max']) + int(search_space_config['INTERMEDIATE_SIZE']['bounds']['step']), int(search_space_config['INTERMEDIATE_SIZE']['bounds']['step']))
        return search_space

def get_subconfig(cand):
    subconfig = dict()
    subconfig['sample_layer_num'] = cand[0]
    subconfig['sample_num_attention_heads'] = [cand[1]] * cand[0]
    subconfig['sample_qkv_sizes'] = [cand[2]] * cand[0]
    subconfig['sample_hidden_size'] = cand[3]
    subconfig['sample_intermediate_sizes'] = [cand[4]] * cand[0]
    return subconfig

def bert_populate_random_func(search_space):
    cand_tuple = list() #[layer_num, [num_attention_heads]*layer_num, [qkv_sizes]*layer_num, hidden_size, [intermediate_sizes]*layer_num]
    dimensions = ['head_num', 'hidden_size', 'ffn_size']
    depth = random.choice(search_space['layer_num'])
    cand_tuple.append(depth)
    for dimension in dimensions:
        if dimension == 'head_num':
            head_num = random.choice(search_space[dimension])
            qkv_size = head_num * 64
            cand_tuple.append(head_num)
            cand_tuple.append(qkv_size)
        elif dimension == 'hidden_size':
            cand_tuple.append(random.choice(search_space['hidden_size']))
        elif dimension == 'ffn_size':
            cand_tuple.append(random.choice(search_space['ffn_size']))

    return tuple(cand_tuple)

def bert_is_legal(cand, vis_dict, params, super_net):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False
    subconfig = get_subconfig(cand)
    super_net.set_sample_config(subconfig)
    n_parameters = super_net.calc_sampled_param_num()
    info['params'] = n_parameters / 10.**6
    if info['params'] > params.max_param_limits:
        return False
    if info['params'] < params.min_param_limits:
        return False
    info['visited'] = True
    return True

def bert_mutation_random_func(m_prob, s_prob, search_space, top_candidates):
    cand = list(random.choice(top_candidates))
    depth, num_heads, qkv_sizes, hidden_size, ffn_sizes = cand[0], cand[1], cand[2], cand[3], cand[4]
    random_s = random.random()
    # depth
    if random_s < s_prob:
        new_depth = random.choice(search_space['layer_num'])
        depth = new_depth
        num_heads = random.choice(search_space['head_num'])
        qkv_sizes = num_heads * 64
        hidden_size = random.choice(search_space['hidden_size'])
        ffn_sizes = random.choice(search_space['ffn_size'])
    random_s = random.random()
    if random_s < m_prob:
        # num_heads
        num_heads = random.choice(search_space['head_num'])
        # qkv_sizes
        qkv_sizes = num_heads * 64
    # hidden_size
    random_s = random.random()
    if random_s < s_prob:
        hidden_size = random.choice(search_space['hidden_size'])
    # ffn_sizes
    random_s = random.random()
    if random_s < s_prob:
        ffn_sizes = random.choice(search_space['ffn_size'])

    result_cand = [depth] + [num_heads] + [qkv_sizes] + [hidden_size] + [ffn_sizes]
    return tuple(result_cand)

def bert_crossover_random_func(top_candidates):
    p1 = random.choice(top_candidates)
    p2 = random.choice(top_candidates)
    max_iters_tmp = 50
    while len(p1) != len(p2) and max_iters_tmp > 0:
        max_iters_tmp -= 1
        p1 = random.choice(top_candidates)
        p2 = random.choice(top_candidates)
    cand = []
    for ind, it in enumerate(zip(p1, p2)):
        if ind == 2:
            continue
        elif ind == 1:
            cand.append(random.choice(it))
            cand.append(cand[-1] * 64)
        else:
            cand.append(random.choice(it))
    return tuple(cand)

def get_bert_latency(model, batch_size, max_seq_length, gpu, infer_cnt):
    if gpu is None:
        device = 'cpu'
    else:
        device = 'cuda'
    input_ids = [9333] * max_seq_length
    input_masks = max_seq_length * [1]
    input_segments = max_seq_length * [0]
    input_ids = torch.tensor([input_ids]*batch_size, dtype=torch.long).to(device)
    input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long).to(device)
    input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long).to(device)

    aver_time = 0.
    model.eval()

    for i in range(int(infer_cnt)):
        start = time.time()
        with torch.no_grad():
            model.forward(input_ids, input_masks, input_segments)

        end = time.time()
        sep = 1000 * (end - start)

        if i == 0:
            continue
        else:
            aver_time += sep / (infer_cnt - 1)
    return aver_time

class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x

class LatencyPredictor(object):
    def __init__(self, feature_norm, lat_norm, ckpt_path, lat_dataset_path='./latency_dataset/lat.tmp', feature_dim=10,
                 hidden_dim=400, hidden_layer_num=3, train_steps=5000, bsz=128, lr=1e-5):
        self.dataset_path = lat_dataset_path
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.ckpt_path = ckpt_path

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.train_steps = train_steps
        self.bsz = bsz

    def train(self):
        for i in range(self.train_steps):
            sample_ind = random.sample(range(len(self.train_x)), k=self.bsz)
            sample_x = [self.train_x[sample_ind[k]] for k in range(self.bsz)]
            sample_y = [self.train_y[sample_ind[k]] for k in range(self.bsz)]

            sample_x_tensor = torch.Tensor(sample_x)
            sample_y_tensor = torch.Tensor(sample_y)

            prediction = self.model(sample_x_tensor).squeeze()

            loss = self.criterion(prediction, sample_y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # validation
            if i % 100 == 0:
                with torch.no_grad():
                    sample_x_tensor = torch.Tensor(self.valid_x)
                    sample_y_tensor = torch.Tensor(self.valid_y)

                    prediction = self.model(sample_x_tensor).squeeze()
                    loss = self.criterion(prediction, sample_y_tensor)
                    print(f"Validation loss at {i} steps: {loss}")

        # test
        with torch.no_grad():
            sample_x_tensor = torch.Tensor(self.test_x)
            sample_y_tensor = torch.Tensor(self.test_y)
            prediction = self.model(sample_x_tensor).squeeze()
            loss = self.criterion(prediction, sample_y_tensor)
            print(f"Predicted latency: {prediction}")
            print(f"Real latency: {self.test_y}")
            print(f"Loss: {loss}")

            print(f"RMSE: {np.sqrt(self.criterion(self.lat_norm*sample_y_tensor, self.lat_norm*prediction))}")
            print(f"MAPD: {torch.mean(torch.abs((sample_y_tensor - prediction) / sample_y_tensor))}")

        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config):
        with torch.no_grad():
            def config_2_feature(config):
                features = []
                features.append(config['sample_hidden_size'])
                features.append(config['sample_layer_num'])
                features.append(sum(config['sample_intermediate_sizes']) /
                                (1.0 * len(config['sample_intermediate_sizes'])))
                features.append(sum(config['sample_qkv_sizes']) / (1.0 * len(config['sample_qkv_sizes'])))
                return features

            features = config_2_feature(config)
            features_norm = np.array(features) / self.feature_norm

            prediction = self.model(torch.Tensor(features_norm)).item() * self.lat_norm

        return prediction

    def split(self):
        sample_num = len(self.dataset['x'])
        train_num = int(np.floor(0.8 * sample_num))
        valid_num = int(np.floor(0.1 * sample_num))

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num+valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num+valid_num)]

        self.test_x = self.dataset['x'][(train_num+valid_num):]
        self.test_y = self.dataset['y'][(train_num+valid_num):]

    def read_dataset(self):
        features_norm_all = []
        lats_all = []
        cnt = 0
        with open(self.dataset_path, 'r') as fid:
            # next(fid) # skip first line of CSV
            for line in fid:
                line = line.strip()

                try:
                    subbert_config, inf_time = line.split('\t')
                    subbert_config = json.loads(json.dumps(ast.literal_eval(subbert_config)))
                except:
                    print('Got error!')

                def config_2_feature(config):
                    features = []
                    features.append(config['sample_hidden_size'])
                    features.append(config['sample_layer_num'])
                    features.append(sum(config['sample_intermediate_sizes']) /
                                    (1.0 * len(config['sample_intermediate_sizes'])))
                    features.append(sum(config['sample_qkv_sizes']) / (1.0 * len(config['sample_qkv_sizes'])))
                    return features

                features_eval = config_2_feature(subbert_config)
                features_norm = np.array(features_eval) / self.feature_norm
                features_norm_all.append(features_norm)
                lats_all.append(float(inf_time) / self.lat_norm)

                cnt += 1
                if cnt % 100000 == 0:
                    print('Loaded {} structures!'.format(cnt))

        tmp = list(zip(features_norm_all, lats_all))
        random.shuffle(tmp)
        features_norm_all, lats_all = zip(*tmp)
        self.dataset = {'x': features_norm_all, 'y': lats_all}