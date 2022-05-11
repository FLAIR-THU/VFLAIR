    def __init__(self, args):
        self.device = args.device
        self.gpu = args.gpu
        self.dataset_name = args.dataset
        self.epochs = args.main_epochs
        self.learning_rate = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        self.num_classes = args.num_class_list[0]
        self.k = args.k
        self.seed = args.seed
        self.backdoor = 1

        self.report_freq = args.report_freq
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.gamma = args.learning_rate_decay_rate # learning rate decay rate
        self.decay_period = args.decay_period
        self.workers = args.worker_thread_number
        # self.grad_clip = args.grad_clip_value
        # self.label_smooth = args.label_smooth
        self.amplify_rate = args.amplify_rate
        # self.amplify_rate_output = args.amplify_rate_output
    
        # self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        self.apply_discrete_gradients = args.apply_discrete_gradients
        self.discrete_gradients_bins = args.discrete_gradients_bins
        # self.discrete_gradients_bound = args.discrete_gradients_bound
        self.discrete_gradients_bound = 1e-3

        # self.apply_ppdl = args.apply_ppdl
        # self.ppdl_theta_u = args.ppdl_theta_u
        # self.apply_gc = args.apply_gc
        # self.gc_preserved_percent = args.gc_preserved_percent
        # self.apply_lap_noise = args.apply_lap_noise
        # self.noise_scale = args.noise_scale
        # self.apply_discrete_gradients = args.apply_discrete_gradients

        self.name = args.exp_res_dir
        self.name = self.name + '{}-{}-{}-{}-{}'.format(
                args.epochs, args.batch_size, args.amplify_rate, args.seed, time.strftime("%Y%m%d-%H%M%S"))
        print(self.name)

    def train(self):
        amplify_rate = self.amplify_rate