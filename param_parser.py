"""Parsing the parameters."""
import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
def parameter_parser():
    parser = argparse.ArgumentParser(description="run.")
    parser.add_argument('--project',
                        default='runs/TN',
                        help='save to project')
    parser.add_argument('--seed',
                        type=int,
                        default=20,
                        help='Random seed')
    parser.add_argument('--name', 
                        default='TN-0.1-1',
                        help='save to project/name')
    parser.add_argument('--pretrain_dir', 
                        default='explainable-recommendation/runs/train/TN-pretrain',
                        help='pretrain')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
    parser.add_argument('--p',
                    default=0.1,
                    help='save to project/name')
    parser.add_argument('--g',
                    default=1,
                    help='save to project/name')
    # Data
    parser.add_argument('--data-adj-mtx',
                        type=str,
                        default='dataset/TN/gn3/graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-node-feats',
                        type=str,
                        default='dataset/TN/gn3/graph_X.csv',
                        help='Graph node features path')
    parser.add_argument('--data-train',
                        type=str,
                        default='dataset/TN/train.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='dataset/TN/val.csv',
                        help='Validation data path')
    parser.add_argument('--data-test',
                        type=str,
                        default='dataset/TN/test.csv',
                        help='Validation data path')
    parser.add_argument('--fused-features',
                        type=str,
                        default='dataset/TN/fused_features.csv',
                        help='Validation data path')
    parser.add_argument('--image-description',
                        type=str,
                        default='dataset/TN/image2review.json',
                        help='Validation data path')
    parser.add_argument('--short-traj-thres',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-feature',
                        type=str,
                        default='timestamp',
                        help='The name of time feature in the data')
    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--geo-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--user-seq-dim',
                        type=int,
                        default=768,
                        help='User embedding dimensions')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=768,
                        help='Category embedding dimensions')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=4,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')

    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')
    parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
    parser.add_argument('--trainoutf', type=str, default='train.txt',
                    help='output file for generated text')
    parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
    return parser.parse_args()
