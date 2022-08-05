import argparse

def configs(args):
    if args.is_training            == None : args.is_training             = True
    if args.device                 == None : args.device                  = 'cuda'

    if args.dataset                == None : args.dataset                 = 'mnist'
    if args.config                 == None : args.config                  = 'mnist'
    if args.data_train_path        == None : args.data_train_path         = 'data/mnist/mnist_train.npy'
    if args.data_test_path         == None : args.data_test_path          = 'data/mnist/mnist_test.npy'
    if args.save_dir               == None : args.save_dir                = 'checkpoints/mnist_predrnn/'
    if args.gen_frm_dir            == None : args.gen_frm_dir             = 'results/mnist_predrnn/'
    if args.input_length           == None : args.input_length            = 10
    if args.total_length           == None : args.total_length            = 20
    if args.img_width              == None : args.img_width               = 64
    if args.img_height             == None : args.img_height              = 64
    if args.img_channel            == None : args.img_channel             = 1

    if args.model_name             == None : args.model_name              = 'predrnn'
    if args.pretrained_model       == None : args.pretrained_model        = ''
    if args.num_hidden             == None : args.num_hidden              = '128, 128, 128, 128'
    if args.filter_size            == None : args.filter_size             = 5
    if args.stride                 == None : args.stride                  = 1
    if args.patch_size             == None : args.patch_size              = 4
    if args.layer_norm             == None : args.layer_norm              = 0
    if args.decouple_beta          == None : args.decouple_beta           = 0.1

    if args.reverse_scheduled_sampling == None : args.reverse_scheduled_sampling = False
    if args.r_sampling_step_1      == None : args.r_sampling_step_1       = 25000
    if args.r_sampling_step_2      == None : args.r_sampling_step_2       = 50000
    if args.r_exp_alpha            == None : args.r_exp_alpha             = 5000

    if args.scheduled_sampling     == None : args.scheduled_sampling      = True
    if args.sampling_stop_iter     == None : args.sampling_stop_iter      = 50000
    if args.sampling_start_value   == None : args.sampling_start_value    = 1.0
    if args.sampling_changing_rate == None : args.sampling_changing_rate  = 0.00002

    if args.lr                     == None : args.lr                      = 1e-3
    if args.reverse_input          == None : args.reverse_input           = True
    if args.batch_size             == None : args.batch_size              = 16
    if args.max_epoches            == None : args.max_epoches             = 100
    if args.num_save_samples       == None : args.num_save_samples        = 20
    if args.num_valid_samples      == None : args.num_valid_samples       = 50
    if args.n_gpu                  == None : args.n_gpu                   = 1

    if args.visual                 == None : args.visual                  = 0
    if args.visual_path            == None : args.visual_path             = './decoupling_visual'

    if args.injection_action       == None : args.injection_action        = 'concat'
    if args.conv_on_input          == None : args.conv_on_input           = 0
    if args.res_on_conv            == None : args.res_on_conv             = 0
    if args.num_action_ch          == None : args.num_action_ch           = 4

    return args
