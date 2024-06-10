import lightning as L
import torch.nn.functional
import torchmetrics
from torch import optim

from lion_pytorch import Lion


class LitBasic_upd_pare(L.LightningModule):
    def __init__(self, model, only_head=False):
        super().__init__()
        self.model = model
        self.only_head = only_head

        self.save_hyperparameters(logger=False)

        self.train_acc = torchmetrics.R2Score()
        self.test_acc = torchmetrics.R2Score()
        self.val_acc = torchmetrics.R2Score()
        
        self.test_acc_2 = torchmetrics.MeanAbsolutePercentageError()
        self.test_acc_3 = torchmetrics.MeanSquaredError()
        self.test_acc_4 = torchmetrics.MeanAbsoluteError()


        self.crit = nn.MSELoss()

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []
        

    def configure_optimizers(self):
        if self.only_head:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.feature_PA.parameters():
                param.requires_grad = True
            for param in self.model.feature_PC.parameters():
                param.requires_grad = True
            for param in self.model.PA_chem.parameters():
                param.requires_grad = True
            for param in self.model.PC_chem.parameters():
                param.requires_grad = True
            for param in self.model.PA_all.parameters():
                param.requires_grad = True
            for param in self.model.PC_all.parameters():
                param.requires_grad = True
            for param in self.model.layers_stack_all.parameters():
                param.requires_grad = True
            
            # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4 , weight_decay=1e-3) 
            
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5
            )

        else:
            for param in self.model.parameters():
                param.requires_grad = True

            
            # optimizer = Lion(self.model.parameters(), lr=1e-8, weight_decay=1e-3)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-7)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5
            )
        # , threshold=1e-5), threshold_mode='max')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },

        }

    def training_step(self, batch, batch_idx):
        x, y, PA_features, PC_features, Common_features, target = batch

        pred = self.model(x, y, PA_features, PC_features, Common_features)
        target = target.view(-1)
        pred = pred.view(-1)
        train_loss = self.crit(pred, target)
        self.log("train_loss", train_loss)
        self.train_acc.update(pred, target)

        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y, PA_features, PC_features, Common_features, target = batch
        
        pred = self.model(x, y, PA_features, PC_features, Common_features)
        target = target.view(-1)
        pred = pred.view(-1)
        val_loss = self.crit(pred, target)
        self.log("val_loss", val_loss)

        self.val_acc.update(pred, target)

    def test_step(self, batch, batch_idx):

        x, y, PA_features, PC_features, Common_features, target = batch

        pred = self.model(x, y, PA_features, PC_features, Common_features)
        target = target.view(-1)
        pred = pred.view(-1)
        test_loss = self.crit(pred, target)
        self.test_acc.update(pred, target)
        self.test_acc_2.update(pred, target)
        self.test_acc_3.update(pred, target)
        self.test_acc_4.update(pred, target)

    def on_train_epoch_end(self):

        acc_value_train = self.train_acc.compute()
        self.log("r2/train", acc_value_train)
        self.train_acc.reset()

    def on_validation_epoch_end(self):

        acc_value_valid = self.val_acc.compute()
        self.log("r2/val", acc_value_valid)
        self.val_acc.reset()

    def on_test_epoch_end(self):

        acc_value_test = self.test_acc.compute()
        self.log("r2/test: ", acc_value_test)
        self.test_acc.reset()
        
        acc_value_test_2 = self.test_acc_2.compute()
        self.log("MAPE/test: ", acc_value_test_2)
        self.test_acc_2.reset()
        
        acc_value_test_3 = self.test_acc_3.compute()
        self.log("MSE/test: ", acc_value_test_3)
        self.test_acc_3.reset()
        
        acc_value_test_4 = self.test_acc_4.compute()
        self.log("MAE/test: ", acc_value_test_4)
        self.test_acc_4.reset()