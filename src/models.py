""" Code for the main model variants. """

import torch
import torch.nn as nn
import pdb

class SimpleEncoderDecoder(nn.Module):
    """
    A Simple EncoderDecoder model
    """
    def __init__(self,
                 agent_encoder,
                 agent_decoder,
                 device,
                 noise_dim=16):

        super(SimpleEncoderDecoder, self).__init__()

        self.agent_encoder = agent_encoder
        self.agent_decoder = agent_decoder
        self.device = device
        self.noise_dim = noise_dim

    def forward(self, src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, stochastic, *argv):
        self.stochastic = stochastic
        agent_encodings = self.encode(src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, *argv)
        decode = self.decode(agent_encodings, decode_rel_pos, decode_start_pos)

        return decode

    def encode(self, src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, *argv):
        # Encode Scene and Past Agent Paths
        src_trajs = src_trajs.permute(1, 0, 2)  # Convert to (Time X Batch X Dim)

        agent_lstm_encodings = self.agent_encoder(src_trajs, src_lens).squeeze(0) # [Total agents in batch X 32]

        # unsorted_src_trajs = src_trajs[:, unsorter, :] # unsorted
        unsorted_agent_lstm_encodings = agent_lstm_encodings[unsorter, :] # unsorted

        # filtered_src_traj = unsorted_src_trajs[:, agent_masks, :] # unsorted & filtered
        filtered_agent_lstm_encodings = unsorted_agent_lstm_encodings[agent_masks, :] # unsorted & filtered

        return filtered_agent_lstm_encodings #, filtered_src_traj

    def decode(self, agent_encodings, decode_rel_pos, decode_start_pos):
        if self.stochastic:
            noise = torch.randn((agent_encodings.shape[0], self.noise_dim), device=self.device)
        else:
            noise = torch.zeros(self.noise_dim, device=self.device)
            noise = noise.repeat(agent_encodings.shape[0], 1)

        fused_noise_encodings = torch.cat((agent_encodings, noise), dim=1)
        decoder_h = fused_noise_encodings.unsqueeze(0)

        # all_agents_last_rel = filtered_src_traj[-1, :, :] - filtered_src_traj[-2, :, :]
        # all_agents_start_pos = filtered_src_traj[-1, :, :]

        predicted_trajs, final_decoder_h = self.agent_decoder(last_pos_rel=decode_rel_pos,
                                                            hidden_state=decoder_h,
                                                            start_pos=decode_start_pos,
                                                            start_vel=None, # all_agents_past_batch[-1, :, :] - all_agents_past_batch[-2, :, :],
                                                            detach_prediction=False)
        predicted_trajs = predicted_trajs.permute(1, 0, 2) # [B X L X 2]

        return predicted_trajs


class SocialPooling(SimpleEncoderDecoder):
    """
    Social Pooling Model
    """
    def __init__(self,
                 agent_encoder,
                 agent_decoder,
                 spatial_encode_agent,
                 spatial_pooling_net,
                 spatial_fetch_agent,
                 device,
                 noise_dim=16):

        super(SocialPooling, self).__init__(
                                      agent_encoder=agent_encoder,
                                      agent_decoder=agent_decoder,
                                      device=device,
                                      noise_dim=noise_dim)

        self.spatial_encode_agent = spatial_encode_agent
        self.spatial_pooling_net = spatial_pooling_net
        self.spatial_fetch_agent = spatial_fetch_agent

    def encode(self, src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, encode_coords, *argv):
        # Encode Scene and Past Agent Paths
        src_trajs = src_trajs.permute(1, 0, 2) # Convert to (Time X Batch X Dim)

        batch_size = argv[0].shape[0]
        agent_lstm_encodings = self.agent_encoder(src_trajs, src_lens).squeeze(0) # [Total agents in batch X 32]

        # unsorted_src_trajs = src_trajs[:, unsorter, :] # unsorted
        unsorted_agent_lstm_encodings = agent_lstm_encodings[unsorter, :] # unsorted

        # filtered_src_traj = unsorted_src_trajs[:, agent_masks, :] # unsorted & filtered
        filtered_agent_lstm_encodings = unsorted_agent_lstm_encodings[agent_masks, :] # unsorted & filtered

        spatial_encoded_agents = self.spatial_encode_agent(batch_size, unsorted_agent_lstm_encodings, encode_coords)

        # Do social pooling on the pooled agents map.
        fused_grid = self.spatial_pooling_net(spatial_encoded_agents)

        # Fetch fused agents states back w.r.t. coordinates from fused map grid:
        decode_coords = encode_coords[agent_masks]
        fused_agent_encodings = self.spatial_fetch_agent(fused_grid, filtered_agent_lstm_encodings, decode_coords)

        return fused_agent_encodings # , filtered_src_traj


class MATF(SimpleEncoderDecoder):
    """
    A Multi Agent Tensor Fusion Model
    """
    def __init__(self,
                 scene_encoder,
                 agent_encoder,
                 agent_decoder,
                 spatial_encode_agent,
                 spatial_pooling_net,
                 spatial_fetch_agent,
                 device,
                 noise_dim=16):

        super(MATF, self).__init__(agent_encoder=agent_encoder,
                                   agent_decoder=agent_decoder,
                                   device=device,
                                   noise_dim=noise_dim)

        self.scene_encoder = scene_encoder
        self.spatial_encode_agent=spatial_encode_agent
        self.spatial_pooling_net=spatial_pooling_net
        self.spatial_fetch_agent=spatial_fetch_agent

    def forward(self, src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, stochastic, encode_coords, scene_images):
        self.stochastic = stochastic
        agent_encodings = self.encode(src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, encode_coords, scene_images)
        decode = self.decode(agent_encodings, decode_rel_pos, decode_start_pos)

        return decode

    def encode(self, src_trajs, src_lens, unsorter, agent_masks, decode_rel_pos, decode_start_pos, encode_coords, scene_images):
        # Encode Scene and Past Agent Paths
        src_trajs = src_trajs.permute(1, 0, 2) # Convert to (Time X Batch X Dim)

        batch_size = scene_images.shape[0]
        scene_encodings = self.scene_encoder(scene_images)
        agent_lstm_encodings = self.agent_encoder(src_trajs, src_lens).squeeze(0) # [Total agents in batch X 32]

        # unsorted_src_trajs = src_trajs[:, unsorter, :] # unsorted
        unsorted_agent_lstm_encodings = agent_lstm_encodings[unsorter, :] # unsorted

        # filtered_src_traj = unsorted_src_trajs[:, agent_masks, :] # unsorted & filtered
        filtered_agent_lstm_encodings = unsorted_agent_lstm_encodings[agent_masks, :] # unsorted & filtered

        spatial_encoded_agents = self.spatial_encode_agent(batch_size, unsorted_agent_lstm_encodings, encode_coords)

        # Concat pooled agents map and scene feature then fuse them
        concat_map = torch.cat((scene_encodings, spatial_encoded_agents), 1)
        fused_grid = self.spatial_pooling_net(concat_map)

        # Fetch fused agents states back w.r.t. coordinates from fused map grid:
        decode_coords = encode_coords[agent_masks]
        fused_agent_encodings = self.spatial_fetch_agent(fused_grid, filtered_agent_lstm_encodings, decode_coords)

        return fused_agent_encodings #, filtered_src_traj


class MATF_Discriminator(MATF):
    """
    A Multi Agent Tensor Fusion Model
    """
    def __init__(self,
                 scene_encoder,
                 agent_encoder,
                 agent_decoder,
                 spatial_encode_agent,
                 spatial_pooling_net,
                 spatial_fetch_agent,
                 device,
                 discriminator,
                 noise_dim=16):

        super(MATF_Discriminator, self).__init__(scene_encoder=scene_encoder,
                                                 agent_encoder=agent_encoder,
                                                 agent_decoder=agent_decoder,
                                                 spatial_encode_agent=spatial_encode_agent,
                                                 spatial_pooling_net=spatial_pooling_net,
                                                 spatial_fetch_agent=spatial_fetch_agent,
                                                 device=device,
                                                 noise_dim=noise_dim)

        self._classifier = discriminator

    def decode(self, agent_encodings, decode_rel_pos, decode_start_pos):
        discriminator_score = self._classifier(agent_encodings)

        return discriminator_score
