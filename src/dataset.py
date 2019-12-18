import os
import pickle
import multiprocessing as mp

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset


RESNET_DIM = 224
RESNET_HALF = 112

MAX_OBSV_LEN = 20
MAX_PRED_LEN = 30


class ParallelSim(object):
    def __init__(self, processes):
        self.pool = mp.Pool(processes=processes)
        self.total_processes = 0
        self.completed_processes = 0
        self.results = []

    def add(self, func, args):
        self.pool.apply_async(func=func, args=args, callback=self.complete)
        self.total_processes += 1

    def complete(self, result_tuple):
        result, flag = result_tuple
        if flag:
            self.results.append(result)
        self.completed_processes += 1
        print('-- processed {:d}/{:d}'.format(self.completed_processes,
                                              self.total_processes), end='\r')

    def run(self):
        self.pool.close()
        self.pool.join()

    def get_results(self):
        return self.results


def argoverse_collate(batch, map_encoding_size=30):
    # batch_i:
    # 1. past_agents_traj : (Num obv agents in batch_i X 20 X 2)
    # 2. past_agents_traj_len : (Num obv agents in batch_i, )
    # 3. future_agents_traj : (Num pred agents in batch_i X 20 X 2)
    # 4. future_agents_traj_len : (Num pred agents in batch_i, )
    # 5. future_agent_masks : (Num obv agents in batch_i)
    # 6. encode_coordinates : (Num obv agents in batch_i X 2)
    # 7. decode_rel_pos: (Num pred agents in batch_i X 2)
    # 8. decode_start_pos: (Num pred agents in batch_i X 2)
    # 9. map_image : (3 X 224 X 224)
    # 10. scene ID: (string)
    # Typically, Num obv agents in batch_i < Num pred agents in batch_i ##

    batch_size = len(batch)
    # past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, encode_coordinates, decode_rel_pos, decode_start_pos, map_image, scene_id = list(zip(*batch))
    past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, encode_coordinates, decode_rel_pos, decode_start_pos, map_image = list(zip(*batch))

    num_past_agents = np.array([len(x) for x in past_agents_traj])
    past_agents_traj = np.concatenate(past_agents_traj, axis=0)
    past_agents_traj_len = np.concatenate(past_agents_traj_len, axis=0)

    num_future_agents = np.array([len(x) for x in future_agents_traj])
    future_agents_traj = np.concatenate(future_agents_traj, axis=0)
    future_agents_traj_len = np.concatenate(future_agents_traj_len, axis=0)

    future_agent_masks = np.concatenate(future_agent_masks, axis=0)

    batch_encode_coordinates = []
    for b_idx, coords in enumerate(encode_coordinates):
        coords_batch = np.insert(np.expand_dims(coords, axis=1), 0, b_idx, axis=1)
        batch_encode_coordinates.append(coords_batch)
    batch_encode_coordinates = np.concatenate(batch_encode_coordinates, axis=0)
    batch_encode_coordinates = np.ravel_multi_index(batch_encode_coordinates.T, (batch_size, map_encoding_size*map_encoding_size))

    decode_rel_pos = np.concatenate(decode_rel_pos, axis=0)
    decode_start_pos = np.concatenate(decode_start_pos, axis=0)

    # scene_id = np.array(scene_id)

    # Sort batch_past_list in the order of decreasing lengths so as to use pack_padded_seq later.
    sorter = np.argsort(past_agents_traj_len)[::-1]
    sorted_past_agents_traj = past_agents_traj[sorter]
    sorted_past_agents_traj_len = past_agents_traj_len[sorter]
    # Unsorter array so we can unsort the sorted agent encoding tensor that went through the encoder LSTM.
    unsorter = np.zeros_like(sorter)
    unsorter[sorter] = np.arange(len(sorter))

    num_past_agents = torch.LongTensor(num_past_agents)
    sorted_past_agents_traj = torch.FloatTensor(sorted_past_agents_traj)
    sorted_past_agents_traj_len = torch.LongTensor(sorted_past_agents_traj_len)
    unsorter = torch.LongTensor(unsorter)
    num_future_agents = torch.LongTensor(num_future_agents)
    future_agents_traj = torch.FloatTensor(future_agents_traj)
    future_agents_traj_len = torch.LongTensor(future_agents_traj_len)
    batch_encode_coordinates = torch.LongTensor(batch_encode_coordinates)

    decode_rel_pos = torch.FloatTensor(decode_rel_pos)
    decode_start_pos = torch.FloatTensor(decode_start_pos)

    future_agent_masks = torch.BoolTensor(future_agent_masks)
    map_image = torch.stack(map_image, dim=0)

    return map_image, future_agent_masks, num_past_agents, sorted_past_agents_traj, sorted_past_agents_traj_len, unsorter, num_future_agents, future_agents_traj, future_agents_traj_len, batch_encode_coordinates, decode_rel_pos, decode_start_pos #, scene_id


class ArgoverseDataset(Dataset):

    def __init__(self, data_dir, map_version, sample_rate=3, min_past_obv_len=2, min_future_obv_len=10, min_future_pred_len=15, max_distance=56, map_encoding_size=30, transform=None, num_workers=None, cache_file=None):
        """
        Args:
        :param data : List of [scene_id, scene_image, number_agents, past_list, future_list,
                               encode_coordinates, decode_coordinates]
        """
        self.sample_rate = sample_rate
        self.min_past_obv_len = min_past_obv_len
        self.min_future_obv_len = min_future_obv_len
        self.min_future_pred_len = min_future_pred_len
        self.max_distance = max_distance

        if map_version=='1.1' or map_version=='1.2' or map_version=='1.3':
            self.map_version = map_version
        else:
            raise("Invalid map: either 1.1, 1.2, or 1.3")
        self.map_encoding_size = map_encoding_size

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        self.num_workers = num_workers

        # Extract Data:
        if cache_file is None:
            self.get_data(data_dir)
        else:
            if os.path.isfile(cache_file):
                self.load_cache(cache_file)
            else:
                self.get_data(data_dir, save_cache_dir=cache_file)

    def __getitem__(self, idx):
        # Create one past list and future list with all the
        past_agents_traj = self.past_agents_traj_list[idx]
        past_agents_traj_len = self.past_agents_traj_len_list[idx]
        future_agents_traj = self.future_agents_traj_list[idx]
        future_agents_traj_len = self.future_agents_traj_len_list[idx]
        future_agent_masks = self.future_agent_masks_list[idx]
        encode_coordinates = self.encode_coordinates[idx]
        decode_rel_pos = self.decode_rel_pos[idx]
        decode_start_pos = self.decode_start_pos[idx]
        scene_id = self.scene_id[idx]

        map_image = Image.open(self.scene_map_paths[idx])
        map_image = self.transform(map_image)

        scene_data = (past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, encode_coordinates, decode_rel_pos, decode_start_pos, map_image) #, scene_id)
        return scene_data

    def __len__(self):
        return len(self.scene_id)

    def load_cache(self, cache_dir):
        with open(cache_dir, 'rb') as f:
            results = pickle.load(f)
        self.past_agents_traj_list, self.past_agents_traj_len_list, self.future_agents_traj_list,\
        self.future_agents_traj_len_list, self.future_agent_masks_list, self.encode_coordinates,\
        self.decode_rel_pos, self.decode_start_pos, self.scene_map_paths, self.scene_id = list(zip(*results))

    def get_data(self, root_dir, save_cache_dir=None):
        print(f'Extracting data from: {root_dir}')

        sub_directories = os.listdir(root_dir)
        sub_directories.sort()
        path_lists = []

        num_dir = len(sub_directories)
        for i, sub_directory in enumerate(sub_directories):
            sub_directory = root_dir + sub_directory + '/'
            path_lists.extend(self.extract_directory(sub_directory))
            print('{:d}/{:d}'.format(i,num_dir), end='\r')

        if self.num_workers:
            num_processes = self.num_workers
        else:
            num_processes = mp.cpu_count()

        runner = ParallelSim(processes=num_processes)

        for path_list in path_lists:
            runner.add(self.extract_submodule_multicore, (path_list, ))

        runner.run()
        results = runner.get_results()

        if save_cache_dir is not None:
            with open(save_cache_dir, 'wb') as f:
                pickle.dump(results, f) 

        self.past_agents_traj_list, self.past_agents_traj_len_list, self.future_agents_traj_list,\
        self.future_agents_traj_len_list, self.future_agent_masks_list, self.encode_coordinates,\
        self.decode_rel_pos, self.decode_start_pos, self.scene_map_paths, self.scene_id = list(zip(*results))

        print('Extraction Compltete!\n')

    def extract_directory(self, directory):

        scene_segments = os.listdir(directory)
        scene_segments.sort(key=lambda x: int(x[-8:], 16))
        path_lists = []

        num_segments = len(scene_segments)
        for i, scene_segment in enumerate(scene_segments):
            observation_dir = directory + scene_segment + '/observation'
            observations = os.listdir(observation_dir)
            observations.sort()
            prediction_dir = directory + scene_segment + '/prediction'
            predictions = os.listdir(prediction_dir)
            predictions.sort()

            # path_lists.append((directory, scene_segment, '019.pkl'))

            assert(len(predictions) == len(observations))
            print('{:d}/{:d}'.format(i,num_segments), end='\r')

            path_lists.append((directory, scene_segment, observations[0]))
            for j in range(0, len(observations), self.sample_rate):
                path_lists.append((directory, scene_segment, observations[j]))

        return path_lists

    def extract_submodule_multicore(self, path_list):

        directory, scene_segment, observation = path_list
        observation_path = directory + scene_segment + '/observation/' + observation
        prediction_path = directory + scene_segment + '/prediction/' + observation

        map_path = directory + scene_segment + '/map/v{:s}/'.format(self.map_version) + observation.replace('pkl', 'png')

        with open(observation_path, 'rb') as f:
            observation_df = pickle.load(f)
        with open(prediction_path, 'rb') as f:
            prediction_df = pickle.load(f)

        past_agent_ids, future_agent_ids_mask = self.get_agent_ids(observation_df, self.min_past_obv_len, self.min_future_obv_len, self.min_future_pred_len)

        past_traj = None
        past_traj_len = None
        future_traj = None
        future_traj_len = None
        encode_coordinate = None
        decode_rel_pos = None
        decode_start_pos = None

        condition = bool(future_agent_ids_mask.sum())
        if condition:
            past_traj, past_traj_len, future_traj, future_traj_len, encode_coordinate, decode_rel_pos, decode_start_pos = self.extract_trajectory_info(observation_df, prediction_df, past_agent_ids, future_agent_ids_mask)

        scene_id = (scene_segment + '/' + observation)
        return (past_traj, past_traj_len, future_traj, future_traj_len, future_agent_ids_mask, encode_coordinate, decode_rel_pos, decode_start_pos, map_path, scene_id), condition


    def get_agent_ids(self, dataframe, min_past_obv_len, min_future_obv_len, min_future_pred_len):
        """
        Returns:
                List of past agent ids: List of agent ids that are to be considered for the encoding phase.
                Future agent ids mask: A mask which dentoes if an agent in past agent ids list is to be considered
                                       during decoding phase.
        """
        # Select past agent ids for the encoding phase.
        past_df = dataframe.loc[((dataframe['class'] == 'VEHICLE') | (dataframe['class'] == 'LARGE_VEHICLE') | (dataframe['class'] == 'PEDESTRIAN') | (dataframe['class'] == 'ON_ROAD_OBSTACLE') | (dataframe['class'] == 'BICYCLE') | (dataframe['class'] == 'AGENT') | (dataframe['class'] == 'AV') | (dataframe['class'] == 'OTHERS'))
                        & (dataframe['observation_length']>=min_past_obv_len)]
        past_agent_ids = past_df['track_id'].unique()

        # Check if the encoding trajectories have their current position in the region of interest.
        updated_past_agent_ids = []
        for agent_id in past_agent_ids:
            last_pos = past_df[past_df['track_id'] == agent_id].iloc[-1][['X','Y']].to_numpy()
            if np.any(np.abs(last_pos) > self.max_distance):
                pass
            else:
                updated_past_agent_ids.append(agent_id)

        updated_past_agent_ids = np.array(updated_past_agent_ids)

        # Select future agent ids for the decoding phase.
        future_df = dataframe.loc[((dataframe['class'] == 'VEHICLE') | (dataframe['class'] == 'LARGE_VEHICLE') | (dataframe['class'] == 'PEDESTRIAN') | (dataframe['class'] == 'BICYCLE') | (dataframe['class'] == 'AGENT') | (dataframe['class'] == 'AV') | (dataframe['class'] == 'OTHERS'))
                          & (dataframe['observation_length']>=min_future_obv_len) & (dataframe['prediction_length']>=min_future_pred_len)]
        future_agent_ids = future_df['track_id'].unique()

        # Create a mask corresponding to the past_agent_ids list where the value '1' in mask denotes
        # that agent is to be considered while decoding and 0 denotes otherwise.
        future_agent_ids_mask = np.isin(updated_past_agent_ids, future_agent_ids)

        return updated_past_agent_ids, future_agent_ids_mask

    def extract_trajectory_info(self, obv_df, pred_df, past_agent_ids, future_agent_ids_mask):
        """
        Extracts the past and future trajectories of the agents as well as the encode and decode
        coordinates.
        """
        past_traj_list = []
        past_traj_len_list = []

        future_traj_list = []
        future_traj_len_list = []

        encode_coordinate_list = []

        decode_rel_pos_list = []
        decode_start_pos_list = []

        for agent_id in past_agent_ids:
            mask = obv_df['track_id'] == agent_id
            past_agent_traj = obv_df[mask][['X', 'Y']].to_numpy().astype(np.float32)

            decode_rel_pos_list.append(past_agent_traj[-1] - past_agent_traj[-2])
            decode_start_pos_list.append(past_agent_traj[-1])

            x, y = past_agent_traj[-1]
            encode_coords = [(2*x+RESNET_HALF) * self.map_encoding_size / RESNET_DIM,
                             (-2*y+RESNET_HALF) * self.map_encoding_size / RESNET_DIM]
            encode_coordinate_list.append(encode_coords)

            obsv_len = past_agent_traj.shape[0]
            obsv_pad = MAX_OBSV_LEN - obsv_len

            if obsv_pad:
                past_agent_traj = np.pad(past_agent_traj, ((0, obsv_pad), (0, 0)), mode='constant')
                past_traj_list.append(past_agent_traj)
                past_traj_len_list.append(obsv_len)
            else:
                past_traj_list.append(past_agent_traj)
                past_traj_len_list.append(obsv_len)

        for agent_id in past_agent_ids[future_agent_ids_mask]:
            mask = pred_df['track_id'] == agent_id
            future_agent_traj = pred_df[mask][['X', 'Y']].to_numpy().astype(np.float32)

            pred_len = future_agent_traj.shape[0]
            pred_pad = MAX_PRED_LEN - pred_len

            if pred_pad:
                future_agent_traj = np.pad(future_agent_traj, ((0, pred_pad), (0, 0)), mode='constant')
                future_traj_list.append(future_agent_traj)
                future_traj_len_list.append(pred_len)
            else:
                future_traj_list.append(future_agent_traj)
                future_traj_len_list.append(pred_len)

        past_traj_list = np.array(past_traj_list)
        past_traj_len_list = np.array(past_traj_len_list)

        future_traj_list = np.array(future_traj_list)
        future_traj_len_list = np.array(future_traj_len_list)

        encode_coordinate_list = np.array(encode_coordinate_list, dtype=np.int64)
        encode_coordinate_list = np.ravel_multi_index(encode_coordinate_list.T, dims=(self.map_encoding_size, self.map_encoding_size))

        decode_rel_pos_list = np.array(decode_rel_pos_list)
        decode_start_pos_list = np.array(decode_start_pos_list)

        return past_traj_list, past_traj_len_list, future_traj_list, future_traj_len_list, encode_coordinate_list, decode_rel_pos_list, decode_start_pos_list
