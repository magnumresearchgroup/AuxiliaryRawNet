import torch
import speechbrain as sb
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from speechbrain import Stage
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.RNN import GRU
from speechbrain.lobes.models.ECAPA_TDNN import Res2NetBlock
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.linear import Linear
from tqdm.contrib import tqdm
from torch.nn import MaxPool1d
from models.BinaryMetricStats import BinaryMetricStats


class RawSNet(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        # wavs, mag, fbanks, mfccs, lens = self.prepare_features(batch.sig,stage)
        wavs, cqt_mag, lens = self.prepare_cqt(batch.sig,stage)

        wavs = torch.unsqueeze(wavs, 2)
        stack_features = []
        stack_features.append(torch.unsqueeze(self.modules.raw_encoder(wavs, lens),2))
        # stack_features.append(self.modules.fbanks_encoder(fbanks, lens))
        stack_features.append(self.modules.fbanks_encoder(cqt_mag, lens))

        enc_output = torch.cat(tuple(stack_features), dim=1)

        enc_output = self.modules.batch_norm(enc_output)
        enc_output = self.modules.conv_1d(enc_output)
        enc_output = enc_output.transpose(1, 2)

        dec_output = self.modules.decoder(enc_output)
        return dec_output

    def prepare_cqt(self, wavs, stage):
        wavs, lens = wavs
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)
        cqt = self.modules.cqt(wavs)
        cqt_mag = torch.sqrt(cqt.pow(2).sum(-1))
        # Add log_cqt here
        cqt_mag = 10.0 * torch.log10(torch.clamp(cqt_mag, min=1e-30))
        cqt_mag -= 10.0

        #Transpose
        cqt_mag = torch.transpose(cqt_mag, 1, 2)

        cqt_mag = self.modules.mean_var_norm(cqt_mag, lens)
        return wavs, cqt_mag, lens

    def prepare_features(self, wavs, stage):
        wavs, lens = wavs
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        mag, fbanks, mfccs = self.modules.mfcc(wavs)



        if fbanks!=None: fbanks = self.modules.mean_var_norm(fbanks, lens)
        if mag!=None:  mag = self.modules.mean_var_norm(mag, lens)
        if mfccs!=None:  mfccs = self.modules.mean_var_norm(mfccs, lens)

        return wavs, mag, fbanks, mfccs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        _, lens = batch.sig
        spkid, _ = batch.key_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            spkid = torch.cat([spkid, spkid], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function
        # loss = sb.nnet.losses.nll_loss(predictions, spkid, lens)
        # loss = sb.nnet.losses.nll_loss(predictions, spkid, lens)

        # predictions_binary =  torch.squeeze(predictions, 1)
        # for i in range(predictions_binary.shape[0]):
        #     predictions_binary[i][0] = -predictions_binary[i][0]
        # predictions_binary = torch.sum(predictions_binary, 1)
        # predictions_binary = torch.unsqueeze(predictions_binary, 1)


        loss = self.hparams.loss_metric(predictions, spkid)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, spkid, lens, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:

            # self.error_metrics.append(batch.id, predictions, spkid, lens)
            # scores =  torch.clone(predictions)

            # scores = torch.squeeze(predictions, 1)
            # for i in range(scores.shape[0]):
            #     scores[i][self.negative_index] = -scores[i][self.negative_index]
            # scores = torch.sum(scores, 1)

            # scores = torch.squeeze(predictions, 1)

            self.error_metrics.append(batch.id, predictions, spkid)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        # self.loss_metric = sb.utils.metric_stats.MetricStats(
        #     metric=sb.nnet.losses.nll_loss
        # )

        # label_encoder = sb.dataio.encoder.CategoricalEncoder()
        # lab_enc_file = os.path.join(self.hparams.save_folder, "label_encoder.txt")
        # label_encoder.load(path=lab_enc_file)
        # positive_index = label_encoder.encode_label('bonafide')
        # self.positive_index = positive_index
        # self.negative_index = label_encoder.encode_label('spoof')
        # sb.nnet.schedulers.update_learning_rate(self.optimizer, 0.00000001)

        self.loss_metric = sb.utils.metric_stats.MetricStats(
            # metric=sb.nnet.losses.nll_loss
            metric=self.hparams.loss_metric
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            # self.error_metrics = self.hparams.error_stats()
            #     self.error_metrics = sb.utils.metric_stats.BinaryMetricStats(
            #         positive_label=positive_index,
            # )
            self.error_metrics = BinaryMetricStats(
                positive_label=1,
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            # wandb.log({"loss": stage_loss})

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                # "error": self.error_metrics.summarize("average"),
                "eer": self.error_metrics.summarize(field='EER'),
                # 'f_score': self.error_metrics.summarize(field='F-score'),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            # old_lr, new_lr = self.hparams.lr_annealing(epoch)
            print('before anchor: %f' % (self.hparams.lr_scheduler.anchor))
            old_lr, new_lr = self.hparams.lr_scheduler([self.optimizer],

                                                       current_epoch = epoch,
                                                       current_loss = stage_loss)
            # old_lr, new_lr = self.hparams.lr_scheduler(stage_loss)
            print('patient counter: %d'%(self.hparams.lr_scheduler.patience_counter))
            print('after anchor: %f'%(self.hparams.lr_scheduler.anchor))
            # print(self.hparams.lr_scheduler.metric_values)

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])
            self.checkpointer.save_and_keep_only(meta=stats,
                                                 # min_keys=["eer"],
                                                 num_to_keep=5,
                                                 keep_recent=True)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def evaluate_batch(self, batch, stage):
        """
        Overwrite evaluate_batch.
        Keep same for stage in (TRAIN, VALID)
        Output probability in TEST stage (from classify_batch)
        """

        if stage != sb.Stage.TEST:

            # Same as before
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
            return loss.detach().cpu()
        else:
            out_prob = self.compute_forward(batch, stage=stage)
            out_prob = out_prob.squeeze(1)
            score, index = torch.max(out_prob, dim=-1)
            # text_lab = self.hparams.label_encoder.decode_torch(index)
            return out_prob, score, index
            # return out_prob, score, index, text_lab

    def evaluate(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        """
        Overwrite evaluate() function so that it can output score file
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0

        """
        added here
        """
        # label_encoder = sb.dataio.encoder.CategoricalEncoder()
        #
        # lab_enc_file = os.path.join(self.hparams.save_folder, "label_encoder.txt")
        #
        # label_encoder.load(path=lab_enc_file)
        #
        # bona_index = label_encoder.encode_label('bonafide')
        # spoof_index = label_encoder.encode_label('spoof')

        pd_out = {'files': [], 'scores': []}

        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                """
                Rewrite here
                bonafide --> 0 , spoof -->
                """
                out_prob, score, index = self.evaluate_batch(batch, stage=Stage.TEST)
                # cm_scores = [out_prob[i][bona_index].item() - out_prob[i][spoof_index].item()
                #              for i in range(out_prob.shape[0])]
                cm_scores = [out_prob[i].item()
                             for i in range(out_prob.shape[0])]
                pd_out['files'] += batch.id
                pd_out['scores'] += cm_scores

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
                """
                Rewrite Over
                """
            pd.DataFrame(pd_out).to_csv('predictions/scores.txt', sep=' ', header=False, index=False)

            # Only run evaluation "on_stage_end" on main process
            # run_on_main(
            #     self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            # )
        self.step = 0

class RawEncoder(torch.nn.Module):
    def __init__(self,
                 device="cpu",
                 activation=torch.nn.LeakyReLU,
                 ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                    Conv1d(
                        in_channels=1,
                        out_channels=128,
                        kernel_size=3,
                        # dilation=3,
                        stride = 3
                    ),
                BatchNorm1d(input_size=128),
                activation(),
            ]
        )
        for _ in range(3):
            self.blocks.extend(
                [ Res2NetBlock(128, 128,  scale=1,
                               # dilation=1,
                               # kernel_size=3
                               ),
                  BatchNorm1d(input_size=128),
                  activation(),
                  Res2NetBlock(128, 128, scale=1,
                               # dilation=1,
                               # kernel_size=3
                               ),
                  BatchNorm1d(input_size=128),
                  activation(),
                  MaxPool1d(3)
                   ]
            )


        self.gru = GRU(
                512,
                input_size= 128,
                dropout= 0.3,
                bias=True,
                # num_layers=2,
            )
        self.linear = Linear(
                input_size=256,
                n_neurons=128,
                bias=True,
                combine_dims=False,
            )



    def forward(self, x, lens=None):
        """Returns the x-vectors.
        Arguments
        ---------
        x : torch.Tensor
        """
        # x = x.transpose(1, 2)
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                if isinstance(layer,MaxPool1d):
                    x = layer(x.permute(0,2,1)).permute(0,2,1)
                else:
                    x = layer(x)
        # x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        # x = self.linear(x)
        return x

class Decoder(sb.nnet.containers.Sequential):
    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1,
    ):
        super().__init__(input_shape=input_shape)

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            # Added here
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )
            self.DNN[block_name].append(activation(), layer_name="act")


        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        # self.append(
        #     sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        # )