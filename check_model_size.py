import torch  # noqa: F401
import speechbrain as sb

from ptflops import get_model_complexity_info
import sys
from hyperpyyaml import load_hyperpyyaml

class test_comlex(torch.nn.Module):
    def __init__(
        self, args
    ):
        super().__init__()
        self.raw_encoder = args['raw_encoder']
        self.decoder = args['decoder']
        self.main_encoder = args['fbanks_encoder']
        self.batch_norm = args['batch_norm']
        self.conv = args['conv']


    def forward(self, x):
        # x -> 95000

        x = self.raw_encoder(x)
        x = torch.unsqueeze(x, 2)
        x1 = torch.randn(1, 500, 40)
        x1 = self.main_encoder(x1)
        x = torch.cat((x, x1), dim=1)
        x = self.batch_norm(x)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)

        return x

hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

# Initialize ddp (useful only for multi-GPU DDP training).
sb.utils.distributed.ddp_init_group(run_opts)

# Load hyperparameters file with command-line overrides.
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

models = {
   'fbanks_encoder': hparams["modules"]['fbanks_encoder'],
    'raw_encoder': hparams["modules"]['raw_encoder'],
    'conv': hparams["modules"]['conv_1d'],
    'batch_norm' : hparams["modules"]['batch_norm'],
    'decoder': hparams["modules"]['decoder'],

}


m = test_comlex(models)

macs_encoder, params = get_model_complexity_info(m, (64000, 1), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
# macs_encoder, params = get_model_complexity_info(m, (500, 40), as_strings=True,
#                                          print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs_encoder))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))



