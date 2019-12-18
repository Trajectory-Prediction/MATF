""" Code for all the model submodules part
    of various model architecures. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class deconv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class AgentEncoderLSTM(nn.Module):

    def __init__(self, device, input_dim=2, embedding_dim=32, h_dim=32, mlp_dim=512, num_layers=1, dropout=0.3):
        super(AgentEncoderLSTM, self).__init__()

        self.device = device
        self._mlp_dim = mlp_dim
        self._h_dim = h_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._input_dim = input_dim

        self._encoder = nn.LSTM(embedding_dim, h_dim, num_layers)
        self._spatial_embedding = nn.Linear(input_dim, embedding_dim)
        self._drop_out = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        # h_0, c_0 of shape (num_layers * num_directions, batch, hidden_size)
        # batch size should be number of agents in the whole batch, instead of number of scenes
        return (
            torch.zeros(self._num_layers, batch_size, self._h_dim).to(self.device),
            torch.zeros(self._num_layers, batch_size, self._h_dim).to(self.device)
        )

    def forward(self, obs_traj, src_lens):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        batch_size = obs_traj.size(1)
        hidden = self.init_hidden(batch_size)

        # convert to relative, as Social GAN do
        rel_curr_ped_seq = torch.zeros_like(obs_traj).to(self.device)
        rel_curr_ped_seq[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

        # Encode observed Trajectory
        obs_traj_embedding = self._spatial_embedding(rel_curr_ped_seq.reshape(-1, self._input_dim))
        obs_traj_embedding = obs_traj_embedding.reshape(-1, batch_size, self._embedding_dim)
        obs_traj_embedding = self._drop_out(obs_traj_embedding)

        obs_traj_embedding = nn.utils.rnn.pack_padded_sequence(obs_traj_embedding, src_lens)
        output, (hidden_final, cell_final) = self._encoder(obs_traj_embedding, hidden)

        output = torch.nn.utils.rnn.pad_packed_sequence(output)

        return hidden_final


class AgentDecoderLSTM(nn.Module):
    '''
    This part of the code is revised from Social GAN's paper for fair comparison
    '''
    def __init__(self, seq_len, device, output_dim = 2, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.3):
        super(AgentDecoderLSTM, self).__init__()

        self._seq_len = seq_len
        self.device = device
        self._embedding_dim = embedding_dim
        self._h_dim = h_dim
        self.num_layers = num_layers
        self._drop_out = nn.Dropout(p=dropout)
        self._decoder = nn.LSTM(embedding_dim, h_dim, num_layers)
        self._spatial_embedding = nn.Linear(output_dim, embedding_dim)
        self._hidden2pos = nn.Linear(h_dim, output_dim)

    def relative_to_abs(self, rel_traj, start_pos=None):
        """
        Inputs:
        - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        # in our case, start pos is always 0
        if start_pos is None:
            start_pos = torch.zeros_like(rel_traj[0]).to(self.device)

        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)

        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos

        return abs_traj.permute(1, 0, 2)

    def forward(self, last_pos_rel, hidden_state, start_pos=None, start_vel=None, detach_prediction=False):
        """
        Inputs:
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch_size = last_pos_rel.size(0)
        decoder_input = self._spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.reshape(1, batch_size, self._embedding_dim)
        decoder_input = self._drop_out(decoder_input)
        
        # State_tuple
        if self.num_layers > 1:
            zero_hidden_states = torch.zeros((self.num_layers-1), hidden_state.shape[1], hidden_state.shape[2]).to(self.device)
            decoder_h = torch.cat((hidden_state, zero_hidden_states), dim=0)
            decoder_c = torch.zeros_like(decoder_h)
            state_tuple = (decoder_h, decoder_c)
        else:
            decoder_c = torch.zeros_like(hidden_state)
            state_tuple = (hidden_state, decoder_c)

        pred_traj_fake_rel = []
        for _ in range(self._seq_len):
            output, state_tuple = self._decoder(decoder_input, state_tuple)
            predicted_rel_pos = self._hidden2pos(output.reshape(batch_size, self._h_dim))
            pred_traj_fake_rel.append(predicted_rel_pos) # [B X 2]

            # For next decode step:
            if detach_prediction:
                decoder_input = self._spatial_embedding(predicted_rel_pos.detach())
            else:
                decoder_input = self._spatial_embedding(predicted_rel_pos)
            decoder_input = decoder_input.reshape(1, batch_size, self._embedding_dim)
            decoder_input = self._drop_out(decoder_input)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0) # [L X B X 2]

        return self.relative_to_abs(pred_traj_fake_rel, start_pos), state_tuple[0]


class AgentsMapFusion(nn.Module):

    def __init__(self, in_channels=32+32, out_channels=32):
        super(AgentsMapFusion, self).__init__()

        self._conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self._pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._conv2 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self._pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._conv3 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=4, stride=1, padding=1, dilation=1)

        self._deconv2 = deconv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                              k_size=4, stride=2, padding=1)

    def forward(self, input_tensor):
        conv1 = self._conv1.forward(input_tensor)
        conv2 = self._conv2.forward(self._pool1.forward(conv1))
        conv3 = self._conv3.forward(self._pool2.forward(conv2))

        up2 = self._deconv2.forward(conv2)
        up3 = F.interpolate(conv3, scale_factor=5)

        features = conv1 + up2 + up3
        return features


class SpatialEncodeAgent(nn.Module):

    def __init__(self, device, spatial_encoding_size=30):
        super(SpatialEncodeAgent, self).__init__()
        self.device = device
        self.spatial_encoding_size = spatial_encoding_size

    def forward(self, batch_size, agent_encodings, encode_coordinates):
        encode_coordinates = encode_coordinates.numpy()
        channel = agent_encodings.shape[-1]
        init_map = torch.zeros((batch_size*self.spatial_encoding_size*self.spatial_encoding_size, channel), device=self.device)

        overlap_idx = set()
        encoding_tensor_dict = {}
        ## step1: collect all encodings according to their indices
        for veh_i, coord_idx in enumerate(encode_coordinates):
            if coord_idx in encoding_tensor_dict.keys():
                overlap_idx.add(coord_idx) # record if overlapping
                encoding_tensor_dict[coord_idx].append(agent_encodings[veh_i])
            else:
                encoding_tensor_dict[coord_idx] = [agent_encodings[veh_i]]

        ## step2: apply max_pool for those encodings that are overlaping
        coord_idx_list = list(encoding_tensor_dict.keys())
        encoding_list = []
        for coord_idx in coord_idx_list:
            if coord_idx in overlap_idx:
                pooled_encoding, _ = torch.max(torch.stack(encoding_tensor_dict[coord_idx], dim=0), dim=0)
                encoding_list.append(pooled_encoding)
            else:
                encoding_list.extend(encoding_tensor_dict[coord_idx])

        ## step3: insert the encodings at the corresponding indices
        encoding_list = torch.stack(encoding_list, dim=0)
        init_map[coord_idx_list] = encoding_list  # NEEDS Review: Before init_map[coord_idx_list] = encoding_list 

        init_map = init_map.reshape((batch_size, self.spatial_encoding_size, self.spatial_encoding_size, channel))
        init_map = init_map.permute((0, 3, 1, 2))

        return init_map


class SpatialFetchAgent(nn.Module):

    def __init__(self, device):
        super(SpatialFetchAgent, self).__init__()
        self.device = device

    def forward(self, fused_grid, filtered_agent_encodings, decode_coordinates):
        # Rearange the fused grid so that linearized index may be used.
        batch, channel, map_h, map_w = fused_grid.shape
        fused_grid = fused_grid.permute((0, 2, 3, 1)) # B x H x W x C
        fused_grid = fused_grid.reshape((batch*map_h*map_w, channel))

        fused_encodings = fused_grid[decode_coordinates, :]
        final_encoding = fused_encodings + filtered_agent_encodings

        return final_encoding


class ResnetShallow(nn.Module):

    def __init__(self, dropout=0.5):  # Output Size: 30 * 30
        super(ResnetShallow, self).__init__()

        self.trunk = models.resnet18(pretrained=True)

        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.upscale4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.shrink = conv2DBatchNormRelu(in_channels=384, n_filters=32,
                                          k_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image):
        x = self.trunk.conv1(image)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)  # /8 the size
        x3 = self.trunk.layer3(x2)  # 16
        x4 = self.trunk.layer4(x3)  # 32

        x3u = self.upscale3(x3)
        x4u = self.upscale4(x4)

        xall = torch.cat((x2, x3u, x4u), dim=1)
        xall = F.interpolate(xall, size=(30, 30))
        final = self.shrink(xall)

        output = self.dropout(final)

        return output


class Classifier(nn.Module):
    def __init__(self, device, embed_dim_agent, classifier_hidden=512, dropout=0.5):
        super(Classifier, self).__init__()
        self._classifier = nn.Sequential(
                nn.Linear(embed_dim_agent, classifier_hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_hidden, 1),
                nn.Sigmoid()
                ).to(device)

    def forward(self, x):
        return self._classifier(x)
