import os
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import SimpleEncoderDecoder, SocialPooling, MATF, MATF_Discriminator
from dataset import ArgoverseDataset, argoverse_collate
from model_utils import AgentEncoderLSTM, AgentDecoderLSTM, AgentsMapFusion, SpatialEncodeAgent, SpatialFetchAgent, ResnetShallow, Classifier
from utils import ModelTrainer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_path = '/mnt/sdb1/shpark/cmu_mmml_project/matf_experiments/' + args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_')

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    print(f"Current Exp Path: {exp_path}")

    logger = SummaryWriter(exp_path + '/logs')

    train_dataset = ArgoverseDataset(args.train_dir, map_version=args.map_version, sample_rate=args.sample_rate,
                                     num_workers=args.num_workers, cache_file=args.train_cache)
    valid_dataset = ArgoverseDataset(args.valid_dir, map_version=args.map_version, sample_rate=args.sample_rate,
                                     num_workers=args.num_workers, cache_file=args.val_cache)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=lambda x: argoverse_collate(x, map_encoding_size=30),
                              num_workers=args.num_workers)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: argoverse_collate(x, map_encoding_size=30),
                              num_workers=1)

    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(valid_dataset)}')

    # Initialize model components:
    __scene_encoder = ResnetShallow(dropout=args.map_dropout)
    __scene_encoder_gan = ResnetShallow(dropout=args.map_dropout)

    for param in __scene_encoder.trunk.parameters():
        if args.freeze_resnet:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for param in __scene_encoder_gan.trunk.parameters():
        if args.freeze_resnet:
            param.requires_grad = False
        else:
            param.requires_grad = True

    __agent_encoder = AgentEncoderLSTM(
                                    device, input_dim=args.agent_indim, embedding_dim=args.agent_embed_dim,
                                    h_dim=args.agent_embed_dim, mlp_dim=args.spatial_embedding_dim, num_layers=args.encoder_LSTM_layers,
                                    dropout=args.encoder_lstm_dropout
                                    )
    __agent_encoder_gan = AgentEncoderLSTM(
                                    device, input_dim=args.agent_indim, embedding_dim=args.agent_embed_dim,
                                    h_dim=args.agent_embed_dim, mlp_dim=args.spatial_embedding_dim, num_layers=args.encoder_LSTM_layers,
                                    dropout=args.encoder_lstm_dropout
                                    )


    __agent_decoder=AgentDecoderLSTM(
                                    seq_len=args.nfuture, device=device, output_dim=args.agent_outdim,
                                    embedding_dim=(args.agent_embed_dim + args.noise_dim),
                                    h_dim=(args.agent_embed_dim + args.noise_dim),
                                    num_layers=args.decoder_LSTM_layers, dropout=args.decoder_lstm_dropout
                                    )

    __spatial_encode_agent=SpatialEncodeAgent(
                                            device=device,
                                            spatial_encoding_size=30
                                            )
    __spatial_encode_agent_gan=SpatialEncodeAgent(
                                            device=device,
                                            spatial_encoding_size=30
                                            )

    __device = device
    __noise_dim = args.noise_dim

    # Configure model based on the selected option:
    if args.model_type == 'SimpleEncoderDecoder':
        model = SimpleEncoderDecoder(agent_encoder=__agent_encoder, agent_decoder=__agent_decoder,
                                     device=__device, noise_dim=__noise_dim)

    elif args.model_type == 'SocialPooling':
        model = SocialPooling(agent_encoder=__agent_encoder, agent_decoder=__agent_decoder,
                              spatial_encode_agent=__spatial_encode_agent,
                              spatial_pooling_net=AgentsMapFusion(in_channels=(args.agent_embed_dim), out_channels=args.agent_embed_dim),
                              spatial_fetch_agent=SpatialFetchAgent(device), device=__device, noise_dim=__noise_dim)

    elif args.model_type == 'MATF':
        model = MATF(scene_encoder=__scene_encoder, agent_encoder=__agent_encoder, agent_decoder=__agent_decoder,
                     spatial_encode_agent=__spatial_encode_agent, spatial_pooling_net=AgentsMapFusion(in_channels=(args.agent_embed_dim+32), out_channels=args.agent_embed_dim),
                     spatial_fetch_agent=SpatialFetchAgent(device), device=__device, noise_dim=__noise_dim)

    elif args.model_type == 'MATF_GAN':
        model = MATF(scene_encoder=__scene_encoder, agent_encoder=__agent_encoder, agent_decoder=__agent_decoder,
                     spatial_encode_agent=__spatial_encode_agent, spatial_pooling_net=AgentsMapFusion(in_channels=(args.agent_embed_dim+32), out_channels=args.agent_embed_dim),
                     spatial_fetch_agent=SpatialFetchAgent(device), device=__device, noise_dim=__noise_dim)

        discriminator = MATF_Discriminator(scene_encoder=__scene_encoder_gan, agent_encoder=__agent_encoder_gan, agent_decoder=None,
                                          spatial_encode_agent=__spatial_encode_agent_gan, spatial_pooling_net=AgentsMapFusion(in_channels=(args.agent_embed_dim+32), out_channels=args.agent_embed_dim),
                                          spatial_fetch_agent=SpatialFetchAgent(device), device=__device,
                                          noise_dim=__noise_dim, discriminator=Classifier(device=device, embed_dim_agent = args.agent_embed_dim))

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    # Send model to Device:
    model = model.to(device)
    # if args.gpu_devices:
    #     model = nn.DataParallel(model, device_ids=eval(args.gpu_devices))

    if args.criterion == 'mseloss':
        criterion = torch.nn.MSELoss(reduction='none')

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=1e-4)

    criterion = criterion.to(device)
    output_log = open(exp_path + '/output_log.txt', 'w')

    if args.model_type == 'MATF_GAN':
        discriminator = discriminator.to(device)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate,
                                     weight_decay=1e-4)
        trainer = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer,
                                exp_path, output_log, logger, device, args.load_ckpt,
                                discriminator, args.gan_weight, args.gan_weight_schedule, optimizer_d)

    else:
        trainer = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer,
                                exp_path, output_log, logger, device, args.load_ckpt)

    trainer.train(args.num_epochs)
    output_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Tag
    parser.add_argument('--tag', type=str, help="Add a tag to the saved folder")

    # Misc Parameters
    parser.add_argument('--train_dir', type=str,
                        default='/mnt/sdb1/shpark/dataset/argoverse/argoverse11/argoverse-forecasting-from-forecasting/train/',
                        help="Train Directory")
    parser.add_argument('--valid_dir', type=str,
                        default='/mnt/sdb1/shpark/dataset/argoverse/argoverse11/argoverse-forecasting-from-forecasting/val/',
                        help="Valid Directory")

#     parser.add_argument('--train_dir', type=str,
#                         default='/mnt/sdb1/shpark/dataset/argoverse/argoverse11/argoverse-forecasting-from-tracking-city-cs/train/',
#                         help="Train Directory")
#     parser.add_argument('--valid_dir', type=str,
#                         default='/mnt/sdb1/shpark/dataset/argoverse/argoverse11/argoverse-forecasting-from-tracking-city-cs/val/',
#                         help="Valid Directory")

    parser.add_argument('--train_cache', type=str, help="")
    parser.add_argument('--val_cache', type=str, help="")
    parser.add_argument('--model_type', type=str, default='MATF', help="SimpleEncoderDecoder | SocialPooling | MATF | MATF_GAN")

    parser.add_argument('--map_version', type=str, default='1.3', help="Map version")
    parser.add_argument('--sample_rate', type=int, default=1, help="")
    
    parser.add_argument('--agent_indim', type=int, default=2, help="Default set to 2 (i.e. x,y cordinates)")
    parser.add_argument('--agent_outdim', type=int, default=2, help="")
    parser.add_argument('--agent_embed_dim', type=int, default=32, help="Encoder and Decoder Embedding Dimension")
    parser.add_argument('--spatial_embedding_dim', type=int, default=512, help="")
    parser.add_argument('--noise_dim', type=int, default=16, help="")

    parser.add_argument('--nfuture', type=int, default=30, help="")
    parser.add_argument('--encoder_LSTM_layers', type=int, default=1, help="")
    parser.add_argument('--decoder_LSTM_layers', type=int, default=1, help="")
    parser.add_argument('--encoder_lstm_dropout', type=float, default=0.1, help="")
    parser.add_argument('--decoder_lstm_dropout', type=float, default=0.1, help="")
    parser.add_argument('--map_dropout', type=float, default=0.5, help="")

    # Training Parameters
    parser.add_argument('--criterion', type=str, default='mseloss', help="Training Criterion")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training the model")
    parser.add_argument('--num_workers', type=int, default=24, help="")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('-load', '--load_ckpt', default=None, help='Load Model Checkpoint')

    parser.add_argument('--gpu_devices', type=str, default='1', help="Use Multiple GPUs for training")
    parser.add_argument('--freeze_resnet', type=int, default=1, help="")

    # It first starts with gan weight = 0.1 and when the training epoch reaches 20, gan weight becomes 0.5 and so on.
    parser.add_argument('--gan_weight', type=float, default=[0.5, 0.7, 1, 1.5, 2.0, 2.5], help="Adversarial Training Alpha")  #[0.5, 0.7, 1, 1.5, 2.0, 2.5]
    parser.add_argument('--gan_weight_schedule', type=float, default=[20, 30, 40, 50, 65, 200], help="Decaying Gan Weight by Epoch")  #[15, 25, 40, 50, 65, 200]
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    main(args)
