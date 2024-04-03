import torch
import numpy as np
import pytorch_lightning as pl


class Discriminator(torch.nn.Module):
    def __init__(self, discriminator_input_size):
        super(Discriminator, self).__init__()

        self.discriminator_input_size = discriminator_input_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.discriminator_input_size, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class DenoisedGenerator(torch.nn.Module):
    def __init__(self, discriminator_input_size):
        super(DenoisedGenerator, self).__init__()

        self.discriminator_input_size = discriminator_input_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(1024, self.discriminator_input_size),
            torch.nn.Tanh(),
        )
        self.convs_1d = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=(9,), stride=(1,), padding=4),
            torch.nn.Tanh(),
            torch.nn.Conv1d(16, 16, kernel_size=(9,), stride=(1,), padding=4),
            torch.nn.Tanh(),
            torch.nn.Conv1d(16, 1, kernel_size=(9,), stride=(1,), padding=4))

    def forward(self, x):
        out = self.model(x)
        out = out.unsqueeze(1)
        out = self.convs_1d(out)
        out = out.squeeze()
        return out


class DenoisedGAN(pl.LightningModule):
    def __init__(self,
                 discriminator_input_size=256,
                 lr=0.0002,
                 b1=0.5,
                 b2=0.999,
                 is_denoised=False):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.preds_stage = {"train": {"g_loss": [], "d_loss": []}}

        self.discriminator_input_size = discriminator_input_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.loss = torch.nn.BCELoss()

        self.generator = DenoisedGenerator(discriminator_input_size)
        self.discriminator = Discriminator(discriminator_input_size)

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.lr,
                                 betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.lr,
                                 betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        optg, optd = self.optimizers()

        noise = torch.normal(0, 1, size=(imgs.shape[0], 100), device=self.device)
        fake_inputs = self.generator(noise)
        fake_outputs = self.discriminator(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1], device=self.device)
        g_loss = self.loss(fake_outputs, fake_targets)
        self.preds_stage['train']['g_loss'].append(g_loss.detach().cpu())
        optg.zero_grad()
        g_loss.backward()
        optg.step()

        real_outputs = self.discriminator(imgs)
        real_label = torch.ones(imgs.shape[0], 1, device=self.device)
        noise = torch.normal(0, 1, size=(imgs.shape[0], 100), device=self.device)
        fake_inputs = self.generator(noise)
        fake_outputs = self.discriminator(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1, device=self.device)
        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)
        d_loss = self.loss(outputs, targets)
        self.preds_stage['train']['d_loss'].append(d_loss.detach().cpu())
        optd.zero_grad()
        d_loss.backward()
        optd.step()

    def on_train_epoch_end(self):
        g_loss = self.preds_stage['train']['g_loss']
        g_loss = torch.stack(g_loss)
        g_loss = np.mean([x.item() for x in g_loss])

        d_loss = self.preds_stage['train']['d_loss']
        d_loss = torch.stack(d_loss)
        d_loss = np.mean([x.item() for x in d_loss])

        metrics = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        self.preds_stage['train']['g_loss'].clear()
        self.preds_stage['train']['d_loss'].clear()
