import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,  
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,  
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,  
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,  
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="ACM", choices=["ACM", "coauthor-cs", "coauthor-phy"],)
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')

    parser.add_argument('--u_lip', type=float, default=0.1, help='upper bound of lipschitz constant')

    parser.add_argument('--model_name', type=str, default="SGC", help='model name', choices=["SGC", "GCN"])
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline'])
    parser.add_argument('--save_dir', '-o', type=str, default='debug')


    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
