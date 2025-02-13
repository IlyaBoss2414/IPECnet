import lightning as L
import torch.nn.functional
import torchmetrics
from torch import optim
from lion_pytorch import Lion
import torch.nn as nn

class LitBasic_upd_pare(L.LightningModule):
    def __init__(self, model, only_head=False):
        super().__init__()
        self.model = model
        self.only_head = only_head

        self.save_hyperparameters(logger=False)

        self.crit = nn.BCEWithLogitsLoss()
        self.crit_val = nn.BCEWithLogitsLoss()
            
        self.train_acc = torchmetrics.AUROC(task="binary")
        self.test_acc = torchmetrics.AUROC(task="binary")
        self.val_acc = torchmetrics.AUROC(task="binary")
        
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_outputs = []
        
        self.preds = []
        self.targets = []
        
        self.preds_before = []

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
            
            
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5
            )

        else:
            for param in self.model.parameters():
                param.requires_grad = True

            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)


            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5
            )
    
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
        
        pred = torch.sigmoid(pred)
        self.train_acc.update(pred, target)

        return train_loss

    def validation_step(self, batch, batch_idx):

        x, y, PA_features, PC_features, Common_features, target = batch
        
        pred = self.model(x, y, PA_features, PC_features, Common_features)
        target = target.view(-1)
        pred = pred.view(-1)

        val_loss = self.crit_val(pred, target)
        self.log("val_loss", val_loss)
        
        pred = torch.sigmoid(pred)
        self.val_acc.update(pred, target)
        
        return val_loss

        
    def test_step(self, batch, batch_idx):
        x, y, PA_features, PC_features, Common_features, target = batch

        pred = self.model(x, y, PA_features, PC_features, Common_features)
        target = target.view(-1)
        pred = pred.view(-1)
        
        test_loss = self.crit(pred, target)
        
        self.preds_before.append(pred.detach())

        # Apply sigmoid to get probabilities in range [0, 1] for AUROC
        pred = torch.sigmoid(pred)
        self.test_acc.update(pred, target)

        # Store predictions and targets
        self.preds.append(pred.detach())  # Detach from computation graph to avoid memory issues
        self.targets.append(target.detach())

        return {"loss": test_loss, "preds": pred, "targets": target}


    def on_train_epoch_end(self):

        acc_value_train = self.train_acc.compute()
        self.log("AUC/train", acc_value_train)
        self.train_acc.reset()
        


    def on_validation_epoch_end(self):

        acc_value_valid = self.val_acc.compute()
        self.log("AUC/val", acc_value_valid)
        self.val_acc.reset()

    def on_test_epoch_start(self):
        # Clear previous predictions and targets at the start of the test epoch
        self.preds.clear()
        self.targets.clear()
        self.preds_before.clear()
    
    def on_test_epoch_end(self):
        # Compute AUROC and log it
        acc_value_test = self.test_acc.compute()
        # auc_test = float(f"{acc_value_test.item():.1f}")
        # print(f"AUC/test: {auc_test}") 

        self.log("AUC/test", acc_value_test)
        
        self.test_acc.reset()

        # Concatenate stored predictions and targets
        test_preds = torch.cat(self.preds, dim=0)
        test_targets = torch.cat(self.targets, dim=0)
        
        pred_before = torch.cat(self.preds_before, dim=0)

        # Clear the lists to free memory
        self.preds.clear()
        self.targets.clear()
        self.preds_before.clear()

        return test_preds, test_targets
