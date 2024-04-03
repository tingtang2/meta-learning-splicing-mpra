import logging
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from tqdm import trange

from data.data_loader import create_dataloaders, get_oracle_splits
from models.keras_pretrained_den import (KerasPretrainedDENPredictor, target_isos)
from models.np import SplicingConvCNP1d, SplicingConvCNPwAttn1d, SplicingConvCNPwHyper1d
from models.resnets import UNet
from trainers.base_trainer import BaseTrainer

from torch.nn import MSELoss


def gaussian_logpdf(inputs, mean, sigma, reduction=None):
    """Gaussian log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(inputs)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')


class PretrainedDenNpTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rep_dim = 64

        self.pretrained_dens = {
            target_iso:
                KerasPretrainedDENPredictor(seed=self.seed,
                                            batch_size=32,
                                            path='/home/tingchen/mpra-project/DEN_splicing_pretrained_models/',
                                            target_iso=target_iso) for target_iso in target_isos
        }
        self.generated_sequence_length = 109

        self.name = 'cnp_x_keras_dens'
        self.criterion = MSELoss(reduction='sum')

    def generate_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        total_sequences = []
        total_labels = []

        for target_iso in target_isos:
            sequences, labels = self.pretrained_dens[target_iso].generate_sequences_and_labels()
            total_sequences.append(sequences)
            total_labels.append(labels)

        return torch.from_numpy(np.concatenate(total_sequences)).to(self.device), torch.from_numpy(np.concatenate(total_labels)).to(self.device)

    def train_epoch(self, loader: DataLoader, generated_sequences: torch.Tensor, generated_labels: torch.Tensor):
        self.model.train()
        running_loss = 0.0
        total_examples = 0

        for batch in loader:
            self.optimizer.zero_grad()
            true_examples, true_labels = batch

            # skip final leftover batch because problems will occur lol
            if true_examples.size(0) != self.batch_size:
                continue

            # switch context and targets?
            pred_dist = self.model(x_c=generated_sequences.view(-1,
                                                                self.generated_sequence_length,
                                                                4),
                                   y_c=generated_labels.to(self.device),
                                   x_t=true_examples.to(self.device))

            # loss = -pred_dist.log_prob(true_labels).sum(-1).mean()
            loss = self.criterion(pred_dist.mean.squeeze(1), true_labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_examples += true_labels.size(0)
        return running_loss / total_examples

    def run_experiment(self):
        if self.use_attention:
            self.model = SplicingConvCNPwAttn1d(inducer_net=UNet(in_channels=5),
                                                r_dim=self.rep_dim,
                                                device=self.device,
                                                context_seq_len=109,
                                                num_samples=1).to(self.device)
        elif self.use_hyper:
            self.model = SplicingConvCNPwHyper1d(inducer_net=UNet(in_channels=5),
                                                 r_dim=self.rep_dim,
                                                 device=self.device,
                                                 context_seq_len=109,
                                                 num_samples=1).to(self.device)
        else:
            self.model = SplicingConvCNP1d(inducer_net=UNet(in_channels=self.rep_dim),
                                           r_dim=self.rep_dim,
                                           device=self.device,
                                           seq_len=109).to(self.device)
        generated_sequences, generated_labels = self.generate_sequences()

        self.X_train, self.y_train, self.X_val, self.y_val = get_oracle_splits(42, num=2)
        self.train_loader, self.val_loader, self.data_dim = create_dataloaders(X_train=self.X_train, y_train=self.y_train, X_test=self.X_val, y_test=self.y_val, device=self.device, batch_size=self.batch_size, test_batch_size=self.batch_size)
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = 1e+5
        early_stopping_counter = 0

        for epoch in trange(1, self.epochs + 1):
            idx = torch.multinomial(torch.ones(generated_sequences.shape[0]), num_samples=self.batch_size)
            sampled_generated_sequences = generated_sequences[idx]
            sampled_generated_labels = generated_labels[idx]

            # pick first sample
            sampled_generated_sequences = sampled_generated_sequences[:, 0, :].squeeze()
            sampled_generated_labels = sampled_generated_labels[:, 0, :].squeeze()

            start_time = timer()
            train_loss = self.train_epoch(loader=self.train_loader,
                                          generated_sequences=sampled_generated_sequences,
                                          generated_labels=sampled_generated_labels)
            end_time = timer()

            val_loss, spearman_res, (pearson_r, _) = self.eval(self.val_loader, generated_sequences=sampled_generated_sequences, generated_labels=sampled_generated_labels)

            log_string = (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, spearman correlation: {spearman_res.correlation:.3f}, pearson correlation: {pearson_r:.3f}, patience:{early_stopping_counter},  "
                f"Epoch time = {(end_time - start_time):.3f}s")
            logging.info(log_string)

            if val_loss < best_val_loss:
                self.save_model(self.name)
                early_stopping_counter = 0
                best_val_loss = val_loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                break

            if early_stopping_counter == 10 or early_stopping_counter == 20:
                generated_sequences, generated_labels = self.generate_sequences()

    def eval(self,
             loader: DataLoader,
             generated_sequences: torch.Tensor,
             generated_labels: torch.Tensor,
             save_plot=False) -> Tuple[torch.Tensor,
                                       object]:
        """evaluates model on given data loader. computes loss and spearman correlation between predictions and labels

        Args:
            loader (DataLoader): data loader containing validation/test data
            save_plot (bool, optional): whether or not to saved correlation plot. Defaults to False.

        Returns:
            torch.Tensor: validation loss
        """
        # sample to match batch size of real examples

        self.model.eval()

        predictions = []
        running_loss = 0

        with torch.no_grad():
            for j, batch in enumerate(loader):

                true_examples, true_labels = batch
                if true_examples.size(0) != self.batch_size:
                    continue

                # switch context and targets?
                pred_dist = self.model(x_c=generated_sequences.view(-1,
                                                                    self.generated_sequence_length,
                                                                    4),
                                       y_c=generated_labels.to(self.device),
                                       x_t=true_examples.to(self.device))
                # running_loss += -pred_dist.log_prob(true_labels).sum(-1).item()
                # predictions.append(pred_dist.base_dist.loc)
                running_loss += self.criterion(pred_dist.mean.squeeze(1), true_labels).item()
                predictions.append(pred_dist.base_dist.mean.squeeze(1))

        if save_plot:
            # TODO: do plotting
            pass

        predictions = torch.cat(predictions, dim=0).squeeze()
        full_labels: torch.Tensor = loader.dataset.proportions[:predictions.size(0)]

        print(f'val loss: {running_loss / predictions.size(0):.5f}, total loss: {running_loss:.5f}')
        print('full_labels', full_labels[:5], full_labels.size())
        print('predictions', predictions[:5], predictions.size())

        return running_loss/predictions.size(0), spearmanr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy()), pearsonr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy())

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')