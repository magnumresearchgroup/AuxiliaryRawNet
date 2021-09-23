import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.RawSNet import RawSNet
from datasets.TrainSpeechDataset import get_dataset
from datasets.EvalSpeechDataset import get_dataset as get_eval_dataset

if __name__ == "__main__":
    TRAIN = True
    RESUME = False
    torch.cuda.empty_cache()
    if TRAIN:

        # Reading command line arguments.
        hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

        # Initialize ddp (useful only for multi-GPU DDP training).
        sb.utils.distributed.ddp_init_group(run_opts)

        # Load hyperparameters file with command-line overrides.
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Create dataset objects "train", "valid", and "test".
        datasets = get_dataset(hparams)

        # Initialize the Brain object to prepare for mask training.
        spk_id_brain = RawSNet(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )



        spk_id_brain.fit(
            epoch_counter=spk_id_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["dev"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )


    else:
        print('Start to Inference')
        hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
        sb.utils.distributed.ddp_init_group(run_opts)
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        hparams_file = os.path.join(hparams['output_folder'], 'hyperparams.yaml')
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        hparams['batch_size'] = 64
        hparams['dataloader_options']['batch_size'] = 16
        datasets = get_eval_dataset(hparams)
        # Initialize the Brain object to prepare for mask training.

        spk_id_brain = RawSNet(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        spk_id_brain.evaluate(
            test_set=datasets["eval"],
            min_key="eer",
            progressbar= True,
            test_loader_kwargs=hparams["dataloader_options"],
        )

