import torch
import math
import numpy as np
from contextlib import nullcontext

def train(net, optimizer, criterion, train_loader, val_loader, epochs, use_autocast, model_name="Akita", plot=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = net.to(device)
    total_step = len(train_loader)
    overall_step = 0
    train_loss_values = []
    validation_values = []
    T = total_step*epochs
    T0 = 2*T//5
    t = 0
    lr = LearningRateScheduler(T0, T)

    for epoch in range(epochs):
        total = 0
        running_loss = 0.0
        model.train(True)
        for i, batch_data in enumerate(train_loader):
            # Set learning rate
            for op_params in optimizer.param_groups:
                op_params['lr'] = lr(t)

            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type) if use_autocast else nullcontext:

                # Forward pass
                if len(batch_data) == 2:
                    X, y = batch_data
                    X = X.to(device).float()
                    y = y.to(device).float()
                    outputs = model(X)
                else:
                    X, c, y = batch_data
                    X = X.to(device).float()
                    y = y.to(device).float()
                    c = c.to(device).float()
                    outputs = model(X, c)

                # Calculate loss
                loss = criterion(outputs, y)

            # Backwards pass
            loss.backward()
            running_loss += loss.item()
            total += y.size(0)
            optimizer.step()
            t += 1

            # Printing
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                      1, epochs, i+1, total_step, loss.item()))
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
            if plot:
                info = {('loss_' + model_name): loss.item()}
        train_loss_values.append(running_loss/total)
        validation_values.append(test(model, criterion, val_loader))
        if epoch+1 == 10:
            torch.save(model.state_dict(), "./model_10.pth")
        if epoch+1 == 15:
            torch.save(model.state_dict(), "./model_15.pth")
        if epoch+1 == 20:
            torch.save(model.state_dict(), "./model_20.pth")
        if running_loss/total <= 1e-3:
            break
    return (train_loss_values, validation_values)


def test(net, criterion, test_loader, use_autocast, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = net.to(device)
    model.eval()
    with torch.no_grad():
        total = 0
        running_loss = 0.0
        for batch_data in test_loader:
            with torch.autocast(device_type=device.type) if use_autocast else nullcontext:
                if len(batch_data) == 2:
                    X, y = batch_data
                    X = X.to(device).float()
                    y = y.to(device).float()
                    outputs = model(X)
                else:
                    X, c, y = batch_data
                    X = X.to(device).float()
                    y = y.to(device).float()
                    c = c.to(device).float()
                    outputs = model(X, c)

                loss = criterion(outputs, y)

            running_loss += loss.item()
            total += y.size(0)

        print(f"Accuracy of the network on the test: {running_loss/total}")
        return running_loss/total
    

class WeightedMSELoss():
    def __init__(self, dim=448):
        x = torch.abs(torch.arange(dim).unsqueeze(0)-torch.arange(dim).unsqueeze(1))
        square_weights = self.diagonal_fun(x)
        self.weights = torch.tensor(
            square_weights[np.triu_indices(x.shape[0], 2)]
        ).unsqueeze(0).unsqueeze(-1)

    def diagonal_fun(self, x, max_weight=36):
        return 1 + max_weight * torch.sin(x/500*torch.pi)

    def to(self, device):
        self.weights = self.weights.to(device)
    
    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        """
        Compute weighted mean square error for prediction.
        Args:
            y_pred: Predicted labels
            y_true: True labels
        """
        return ((y_true - y_pred)**2 * self.weights).mean()/y_true.shape[0]
    

class LearningRateScheduler:
    def __init__(self, T0:int, T:int, eta:float=1e-3, C1:float=1e-4, C2:float=1e-6):
        self.T0 = T0
        self.T = T
        self.C1 = C1
        self.C2 = C2
        self.eta = eta
    
    def __call__(self, t:int):
        if t <= self.T0:
            return self.C1 + self.eta
        else:
            f = (t - self.T0) / (self.T - self.T0)
            return self.C2 + (self.eta * math.cos(f * math.pi/2))