import time
import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


def print_out(string, file_front):
    print(string)
    print(string, file=file_front)


class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, criterion,
                 optimizer, exp_path, text_logger, logger, device, load_ckpt=None, discriminator=None, gan_weight=None, gan_weight_schedule=None, optimizer_d=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=(1/2), verbose=True, patience=3)
        self.exp_path = exp_path
        self.text_logger = text_logger
        self.logger = logger
        self.device = device
        self.discriminator = discriminator
        self.stochastic = False

        if self.discriminator is not None:
            self.discriminator = discriminator
            self.gan_weight = gan_weight
            self.gan_weight_schedule = gan_weight_schedule
            self.adversarial_loss = torch.nn.BCELoss()
            self.optimizer_D = optimizer_d
            self.stochastic = True

        if load_ckpt:
            self.load_checkpoint(load_ckpt)

        # Other Parameters
        self.best_valid_ade = None
        self.best_valid_fde = None
        self.start_epoch = 1

    def train(self, num_epochs):
        print_out('TRAINING .....', self.text_logger)

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            print_out("==========================================================================================", self.text_logger)

            if self.discriminator is None:
                train_loss, train_ade, train_fde = self.train_single_epoch()
            else:
                train_g_loss, train_d_loss, train_ade, train_fde  = self.train_gan_single_epoch(epoch)

            valid_ade, valid_fde, scheduler_metric = self.inference()
            self.scheduler.step(scheduler_metric)

            print_out("------------------------------------------------------------------------------------------", self.text_logger)
            if self.discriminator is None:
                print_out(f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} | Train ADE: {train_ade:0.4f} | Train FDE: {train_fde:0.4f}', self.text_logger)
            else:
                print_out(f'| Epoch: {epoch:02} | Train G_Loss: {train_g_loss:0.6f} | Train D_Loss: {train_d_loss:0.6f} |Train ADE: {train_ade:0.4f} | Train FDE: {train_fde:0.4f}', self.text_logger)

            print_out(f'| Epoch: {epoch:02} | Valid ADE: {valid_ade:0.4f} | Valid FDE: {valid_fde:0.4f} | Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n', self.text_logger)

            self.save_checkpoint(epoch, ade=valid_ade, fde=valid_fde)

            # Log values to Tensorboard
            if self.discriminator is None:
                self.logger.add_scalar('data/Train Loss', train_loss, epoch)
                self.logger.add_scalar('data/Learning Rate', self.get_lr(), epoch)
            else:
                self.logger.add_scalar('data/Train G_Loss', train_g_loss, epoch)
                self.logger.add_scalar('data/Train D_Loss', train_d_loss, epoch)
                self.logger.add_scalar('data/G_Learning Rate', self.get_lr(), epoch)
                self.logger.add_scalar('data/D_Learning Rate', self.get_D_lr(), epoch)

            self.logger.add_scalar('data/Train ADE', train_ade, epoch)
            self.logger.add_scalar('data/Train FDE', train_fde, epoch)
            self.logger.add_scalar('data/Scheduler Metric', scheduler_metric, epoch)

            self.logger.add_scalar('data/Valid ADE', valid_ade, epoch)
            self.logger.add_scalar('data/Valid FDE', valid_fde, epoch)

        self.logger.close()
        print_out("Training Complete! ", self.text_logger)

    def train_single_epoch(self):
        """Trains the model for a single round."""

        self.model.train()
        epoch_loss = 0.0
        epoch_ade, epoch_fde = 0.0, 0.0
        epoch_agents = 0.0
        for b, batch in enumerate(self.train_loader):
            print("Working on batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')
            self.optimizer.zero_grad()
            scene_images, agent_masks, num_src_trajs, src_trajs, src_lens, unsorter, num_tgt_trajs, tgt_trajs, tgt_lens, encode_coords, decode_rel_pos, decode_start_pos = batch

            scene_images = scene_images.to(self.device, non_blocking=True)
            src_trajs = src_trajs.to(self.device, non_blocking=True)
            src_lens = src_lens.to(self.device, non_blocking=True)
            tgt_trajs = tgt_trajs.to(self.device, non_blocking=True)
            decode_rel_pos = decode_rel_pos.to(self.device, non_blocking=True)
            decode_start_pos = decode_start_pos.to(self.device, non_blocking=True)

            predicted_trajs = self.model(src_trajs, src_lens, unsorter, agent_masks,
                                         decode_rel_pos[agent_masks], decode_start_pos[agent_masks],
                                         self.stochastic, encode_coords, scene_images)

            # Calculate the sample indices
            with torch.no_grad():
                time_normalizer = []
                all_agent_time_index, all_agent_final_time_index = [], []

                for i, tgt_len in enumerate(tgt_lens):
                    idx_i = np.arange(tgt_len) + i*30
                    normalizer_i = torch.ones(tgt_len) * tgt_len
                    time_normalizer.append(normalizer_i)
                    all_agent_time_index.append(idx_i)
                    all_agent_final_time_index.append(idx_i[-1])

                time_normalizer = torch.cat(time_normalizer).to(self.device)
                all_agent_time_index = np.concatenate(all_agent_time_index)

            # Loss
            batch_loss = self.criterion(predicted_trajs, tgt_trajs)
            batch_loss = batch_loss.reshape((-1, 2))
            batch_loss = batch_loss[all_agent_time_index]
            batch_loss /= torch.unsqueeze(time_normalizer, dim=1)
            batch_loss = batch_loss.sum()/(len(tgt_lens) * 2.0)

            with torch.no_grad():
                error = predicted_trajs - tgt_trajs
                sq_error = (error ** 2).sum(2).sqrt()
                sq_error = sq_error.reshape((-1))

                # ADE
                batch_ade = sq_error[all_agent_time_index]
                batch_ade /= time_normalizer
                batch_ade = batch_ade.sum()

                # FDE
                batch_fde = sq_error[all_agent_final_time_index]
                batch_fde = batch_fde.sum()

            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_ade += batch_ade.item()
            epoch_fde += batch_fde.item()
            epoch_agents += len(tgt_lens)

        epoch_loss /= (b+1)
        epoch_ade /= epoch_agents
        epoch_fde /= epoch_agents
        return epoch_loss, epoch_ade, epoch_fde

    def train_gan_single_epoch(self, epoch):
        """Trains the model for a single round."""

        self.model.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        epoch_ade, epoch_fde = 0.0, 0.0
        epoch_agents = 0.0

        for i, e in enumerate(self.gan_weight_schedule):
            if epoch <= e:
                gan_weight = self.gan_weight[i]
                break

        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            print("Working on batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')
            scene_images, agent_masks, num_src_trajs, src_trajs, src_lens, unsorter, num_tgt_trajs, tgt_trajs, tgt_lens, encode_coords, decode_rel_pos, decode_start_pos = batch

            scene_images = scene_images.to(self.device, non_blocking=True)
            src_trajs = src_trajs.to(self.device, non_blocking=True)
            src_lens = src_lens.to(self.device, non_blocking=True)
            tgt_trajs = tgt_trajs.to(self.device, non_blocking=True)
            decode_rel_pos = decode_rel_pos.to(self.device, non_blocking=True)
            decode_start_pos = decode_start_pos.to(self.device, non_blocking=True)

            predicted_trajs = self.model(src_trajs, src_lens, unsorter, agent_masks,
                                        decode_rel_pos[agent_masks], decode_start_pos[agent_masks],
                                        self.stochastic, encode_coords, scene_images)

            # src_trajs, src_lens are sorted by len
            unsorted_src_trajs = src_trajs[unsorter]
            unsorted_src_lens = src_lens[unsorter]

            tgt_idx = 0
            total_len = []
            true_trajs, fake_trajs = [], []
            # true_trajs are [past + future labels] and fake_trajs are [past + future predicitons]

            for i, mask in enumerate(agent_masks):
                past_len = unsorted_src_lens[i]
                past = unsorted_src_trajs[i, :past_len, :]  # [seq_len X 2]

                if mask: # decode
                    future_len = tgt_lens[tgt_idx]
                    real = tgt_trajs[tgt_idx][:future_len]
                    fake = predicted_trajs[tgt_idx][:future_len]

                    total_len.append(past_len + future_len)
                    true_trajs.append(torch.cat((past, real), axis=0))
                    fake_trajs.append(torch.cat((past, fake), axis=0))
                    tgt_idx+=1

                else: # not decode
                    total_len.append(past_len)
                    true_trajs.append(past)
                    fake_trajs.append(past)

            total_len = torch.stack(total_len)
            true_trajs = np.array(true_trajs)
            fake_trajs = np.array(fake_trajs)

            # padding
            MAX_OBSV_LEN = 20
            MAX_PRED_LEN = 30
            MAX_ALL_LEN = MAX_OBSV_LEN + MAX_PRED_LEN

            padded_true_trajs = []
            padded_fake_trajs = []
            for true_traj, fake_traj in zip(true_trajs, fake_trajs):
                obsv_len = true_traj.shape[0]
                obsv_pad = MAX_ALL_LEN - obsv_len

                if obsv_pad > 0:
                    true_traj = F.pad(input=true_traj, pad=(0, 0, 0, obsv_pad), mode='constant', value=0)
                    fake_traj = F.pad(input=fake_traj, pad=(0, 0, 0, obsv_pad), mode='constant', value=0)
                    padded_true_trajs.append(true_traj)
                    padded_fake_trajs.append(fake_traj)
                else:
                    padded_true_trajs.append(true_traj)
                    padded_fake_trajs.append(fake_traj)

            padded_true_trajs = torch.stack(padded_true_trajs)  # [num_agents X MAX_ALL_LEN X 2]
            padded_fake_trajs = torch.stack(padded_fake_trajs)

            # sorter
            all_sorter = torch.argsort(total_len, descending=True)
            sorted_padded_true_trajs = padded_true_trajs[all_sorter]
            sorted_padded_fake_trajs = padded_fake_trajs[all_sorter]
            total_len = total_len[all_sorter]
            all_unsorter = torch.argsort(all_sorter)
            all_agent_masks = torch.ones_like(agent_masks)

            true_score = self.discriminator(sorted_padded_true_trajs, total_len, all_unsorter, all_agent_masks,
                                            None, None, None, encode_coords, scene_images)  # [num_agents X 1]

            fake_score = self.discriminator(sorted_padded_fake_trajs, total_len, all_unsorter, all_agent_masks,
                                            None, None, None, encode_coords, scene_images)  # [num_agents X 1]

            # Train Generator (i.e. MATF decoder)
            self.model.require_grad = True
            self.discriminator.require_grad = False

            # Calculate the sample indices
            with torch.no_grad():
                time_normalizer = []
                all_agent_time_index, all_agent_final_time_index = [], []

                for i, tgt_len in enumerate(tgt_lens):
                    idx_i = np.arange(tgt_len) + i*30
                    normalizer_i = torch.ones(tgt_len) * tgt_len
                    time_normalizer.append(normalizer_i)
                    all_agent_time_index.append(idx_i)
                    all_agent_final_time_index.append(idx_i[-1])

                time_normalizer = torch.cat(time_normalizer).to(self.device)
                all_agent_time_index = np.concatenate(all_agent_time_index)

            batch_regression_loss = self.criterion(predicted_trajs, tgt_trajs)
            batch_regression_loss = batch_regression_loss.reshape((-1, 2))
            batch_regression_loss = batch_regression_loss[all_agent_time_index]
            batch_regression_loss /= torch.unsqueeze(time_normalizer, dim=1)
            batch_regression_loss = batch_regression_loss.sum()/(len(tgt_lens) * 2.0)

            with torch.no_grad():
                error = predicted_trajs - tgt_trajs
                sq_error = (error ** 2).sum(2).sqrt()
                sq_error = sq_error.reshape((-1))

                # ADE
                batch_ade = sq_error[all_agent_time_index]
                batch_ade /= time_normalizer
                batch_ade = batch_ade.sum()

                # FDE
                batch_fde = sq_error[all_agent_final_time_index]
                batch_fde = batch_fde.sum()

            batch_adversarial_loss = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
            batch_g_loss = batch_regression_loss + (batch_adversarial_loss * gan_weight)

            batch_g_loss.backward(retain_graph=True)
            self.optimizer.step()

            # Train Discriminator
            self.discriminator.require_grad = True
            self.model.require_grad = False

            real_loss = self.adversarial_loss(true_score, torch.ones_like(true_score))
            fake_loss = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
            batch_d_loss = gan_weight*(real_loss + fake_loss)

            batch_d_loss.backward()
            self.optimizer_D.step()

            epoch_g_loss += batch_g_loss.item()
            epoch_d_loss += batch_d_loss.item()
            epoch_ade += batch_ade
            epoch_fde += batch_fde
            epoch_agents += len(tgt_lens)
            torch.cuda.empty_cache()

        epoch_g_loss /= (b+1)
        epoch_d_loss /= (b+1)
        epoch_ade /= epoch_agents
        epoch_fde /= epoch_agents

        return epoch_g_loss, epoch_d_loss, epoch_ade, epoch_fde

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        
        with torch.no_grad():
            epoch_ade, epoch_fde = 0.0, 0.0
            epoch_agents = 0.0

            for b, batch in enumerate(self.valid_loader):
                scene_images, agent_masks, num_src_trajs, src_trajs, src_lens, unsorter, num_tgt_trajs, tgt_trajs, tgt_lens, encode_coords, decode_rel_pos, decode_start_pos = batch
                scene_images = scene_images.to(self.device, non_blocking=True)
                src_trajs = src_trajs.to(self.device, non_blocking=True)
                src_lens = src_lens.to(self.device, non_blocking=True)
                tgt_trajs = tgt_trajs.to(self.device, non_blocking=True)
                decode_rel_pos = decode_rel_pos.to(self.device, non_blocking=True)
                decode_start_pos = decode_start_pos.to(self.device, non_blocking=True)

                # Prediction
                predicted_trajs = self.model(src_trajs, src_lens, unsorter, agent_masks,
                                            decode_rel_pos[agent_masks], decode_start_pos[agent_masks],
                                            self.stochastic, encode_coords, scene_images)

                # Calculate the sample indices
                time_normalizer = []
                all_agent_time_index = []
                all_agent_final_time_index = []

                for i, tgt_len in enumerate(tgt_lens):
                    idx_i = np.arange(tgt_len) + i*30
                    normalizer_i = torch.ones(tgt_len) * tgt_len
                    time_normalizer.append(normalizer_i)
                    all_agent_time_index.append(idx_i)
                    all_agent_final_time_index.append(idx_i[-1])

                time_normalizer = torch.cat(time_normalizer).to(self.device)
                all_agent_time_index = np.concatenate(all_agent_time_index)

                error = predicted_trajs - tgt_trajs
                sq_error = (error ** 2).sum(2).sqrt()
                sq_error = sq_error.reshape((-1))

                # ADE
                batch_ade = sq_error[all_agent_time_index]
                batch_ade /= time_normalizer
                batch_ade = batch_ade.sum()

                # FDE
                batch_fde = sq_error[all_agent_final_time_index]
                batch_fde = batch_fde.sum()

                epoch_ade += batch_ade.item()
                epoch_fde += batch_fde.item()
                epoch_agents += len(tgt_lens)

            epoch_ade /= epoch_agents
            epoch_fde /= epoch_agents

        scheduler_metric = (epoch_ade + epoch_fde) / 2.0

        return epoch_ade, epoch_fde, scheduler_metric

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_D_lr(self):
        for param_group in self.optimizer_D.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_ade': ade,
            'val_fde': fde,
        }

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, ade, fde)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        print_out("Loading checkpoint from {:s}".format(ckpt), self.text_logger)
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.start_epoch = checkpoint['epoch']
