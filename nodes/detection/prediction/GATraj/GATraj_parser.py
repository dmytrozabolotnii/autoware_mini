import argparse
import ast

def get_args():
    parser = argparse.ArgumentParser(description='GATraj')
    parser.add_argument(
        '--ifvalid',default=True,type=ast.literal_eval,
        help="=False,use all train set to train,"
             "=True,use train set to train and valid")
    parser.add_argument(
        '--mlp_decoder',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_offset',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--input_position',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_mix',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--ifGaussian',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--SR',default=True,type=ast.literal_eval)
    parser.add_argument("--pass_time",
                        default=2,
                        type=int)
    parser.add_argument("--final_mode",
                        default=20,
                        type=int)
    parser.add_argument("--T_max",
                        default=1000,
                        type=int)
    parser.add_argument("--eta_min",
                        default=3e-6,
                        type=float)
    parser.add_argument(
        '--output_size',default=2,type=int)
    parser.add_argument(
        '--input_size',default=2,type=int)
    parser.add_argument(
        '--min_obs',default=8,type=int)
    parser.add_argument('--num_pred', type=int, default=1, help='This is the number of predictions for each agent')
    parser.add_argument('--ratio', type=float, default=0.95, help='The overlap ratio of coexisting for group detection')
    parser.add_argument('--z_dim', type=int, default=32, help='This is the size of the latent variable')
    parser.add_argument('--hidden_size', type=int, default=64, help='The size of LSTM hidden state')
    parser.add_argument('--x_encoder_layers', type=int, default=3, help='Number of transformer block layers for x_encoder')
    parser.add_argument('--x_encoder_head', type=int, default=8, help='Head number of x_encoder')
    parser.add_argument(
        '--gpu', default=0,type=int,
        help='gpu id')
    parser.add_argument(
        '--using_cuda',default=True,type=ast.literal_eval) # We did not test on cpu
    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set',default=1,type=int,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument(
        '--base_dir',default='.',
        help='Base directory including these scrits.')
    parser.add_argument(
        '--save_base_dir',default='./savedata/',
        help='Directory for saving caches and models.')
    parser.add_argument(
        '--phase', default='train',
        help='Set this value to \'train\' or \'test\'')
    parser.add_argument(
        '--GT',default=True)
    parser.add_argument(
        '--train_model', default='GATraj',
        help='Your model name')
    parser.add_argument(
        '--load_model', default=0,type=int,
        help="load model weights from this index before training or testing")
    parser.add_argument(
        '--model', default='models.GATraj')
    ######################################
    parser.add_argument(
        '--dataset',default='eth5')
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')
    parser.add_argument(
        '--val_fraction',default=0,type=float)
    #Perprocess
    parser.add_argument(
        '--seq_length',default=20,type=int)
    parser.add_argument(
        '--obs_length',default=8,type=int)
    parser.add_argument(
        '--pred_length',default=12,type=int)
    parser.add_argument(
        '--batch_size',default=64,type=int)
    parser.add_argument(
        '--show_step',default=40,type=int)
    parser.add_argument(
        '--step_ratio',default=0.5,type=int)
    parser.add_argument(
        '--lr_step',default=20,type=int)
    parser.add_argument(
        '--num_epochs',default=1000,type=int)
    parser.add_argument(
        '--ifshow_detail',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--randomRotate',default=True,type=ast.literal_eval,
        help="=True:random rotation of each trajectory fragment")
    parser.add_argument(
        '--neighbor_thred',default=10,type=int)
    parser.add_argument(
        '--learning_rate',default=1e-04,type=float)
    parser.add_argument(
        '--clip',default=1,type=int)

    args, _ = parser.parse_known_args()

    return args
