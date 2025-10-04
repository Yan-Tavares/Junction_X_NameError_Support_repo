import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class CustomMLP_clf(nn.Module):
    def __init__(self, input_dim = 3 , hidden_layer_sizes = [3,3], output_dim = 1, dropout_frac = 0.2):
        super(CustomMLP_clf, self).__init__()
        layers = []
        prev_dim = input_dim

        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_frac))
            prev_dim = layer_size

        layers.append(nn.Linear(prev_dim, output_dim))

        # Softmax for multi-class or sigmoid for binary
        if output_dim == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)

def create_batches(X, y, batch_size):

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for start_idx in range(0, len(X), batch_size):

        end_idx = min(start_idx + batch_size, len(X))

        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]

def prepare_clf_data_and_split( df, y_class_index =  -1, normalize = True,  train_frac = 0.5, val_frac = 0.25 ):

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(data.iloc[:, y_class_index])

    X = df.drop(df.columns[y_class_index], axis=1).values
    y = y_int

    print(f'Dataset loaded')
    print(f'Number of data points: {len(X)}')

    if normalize:
        # Normalize X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split into train and temp (temp will be split into val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= train_frac, random_state=42, stratify=y)

    # train_size + test_size + val_size = 1
    # test_frac = 1 - val_frac - train_frac 
    # test_frac/val_frac = (1 - val_frac)/train_frac - 1

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= (1 - val_frac)/train_frac - 1, random_state=42, stratify=y_temp)

    print(f'Fraction of train data:  {train_frac}')
    print(f'Fraction of val data: {val_frac}')
    print(f'Test/Val split: {(1 - val_frac)/train_frac - 1}')

    return X, y_int, X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model,
                optimizer,
                loss_fn,
                max_epochs,
                max_tolerance,
                val_freq,
                lasso_reg = 0,
                batch_size = 30):

    best_model = MLP_clf.state_dict()

    min_loss = np.inf
    epoch = 0
    tolerance = 0

    batch_losses = []
    val_losses = []
    epoch_numbers = []

    for epoch in range(max_epochs):
        
        
        batch = 0
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size= batch_size):
            

            y_pred = MLP_clf(X_batch)
            batch_loss = loss_fn(y_pred, y_batch)

            l1_contribution = lasso_reg * sum(p.abs().sum() for p in MLP_clf.parameters())
            batch_loss = loss_fn(y_pred, y_batch) + l1_contribution

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            print(f'\rEpoch: {epoch} | Batch: {batch} | Batch Loss: {batch_loss.item():.4f}', end='')

            batch += 1

        if (epoch == 10) or ( epoch% val_freq) == 0:

            with torch.no_grad():
                val_pred = MLP_clf(X_val)
                val_loss = loss_fn(val_pred, y_val)

            epoch_numbers.append(epoch)
            batch_losses.append(batch_loss.item())
            val_losses.append(val_loss.item())

            print(f'\nValidation loss:{val_loss.item():.4f}')

            if val_loss <= min_loss:
                print('Minimal loss achieved')
                min_loss = val_loss
                tolerance = 0
                best_model = MLP_clf.state_dict()
                # torch.save(best_model, 'best_MLP_clf.pth')
                # print('Model saved')

            else: 
                tolerance += 1
            
            print('\n------------------ ')

        if tolerance > max_tolerance:
            print('Maximum tolerance reached')
            break
        
        epoch += 1

    return  best_model, epoch_numbers, val_losses, batch_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('support files/sample database/classifier_cluster_sample_data.csv')

# One-hot encode the last column (target variable)

X, y_int, X_train, X_val, X_test, y_train, y_val, y_test = prepare_clf_data_and_split( 
                                                            data,
                                                            y_class_index =  -1, 
                                                            normalize = True,  
                                                            train_frac = 0.5, 
                                                            val_frac = 0.25 )

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)


MLP_clf = CustomMLP_clf(input_dim = X_train.shape[1], 
                        hidden_layer_sizes = [128, 64, 32], 
                        output_dim = len(np.unique(y_int)),
                        dropout_frac = 0.1).to(device)


best_model, epoch_numbers, val_losses, batch_losses = train_model(
                                                    model = MLP_clf,
                                                    optimizer = torch.optim.AdamW(MLP_clf.parameters(), lr=1e-3, weight_decay=1e-6),
                                                    loss_fn = nn.CrossEntropyLoss(),
                                                    max_epochs = 10000,
                                                    max_tolerance = 20,
                                                    val_freq = 25,
                                                    lasso_reg = 0,
                                                    batch_size = 50)

MLP_clf.load_state_dict(best_model)
print('Loaded best model for test evaluation.')

with torch.no_grad():
    test_pred = MLP_clf(X_test)
    predicted_classes = torch.argmax(test_pred, dim=1).cpu().numpy()
    true_classes = y_test.cpu().numpy()
    test_accuracy = accuracy_score(true_classes, predicted_classes)

print(f'Test Accuracy: {test_accuracy:.4f}')

plt.plot(epoch_numbers,val_losses, color = 'green')
plt.plot(epoch_numbers,batch_losses,color = 'red')
plt.grid()
plt.show()


# from sklearn.neural_network import MLPClassifier

# mlp_sklearn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=10000, random_state=42)
# mlp_sklearn.fit(np.array(X_train.cpu()), np.array(y_train.cpu()))
# mlp_pred = mlp_sklearn.predict(X_test.cpu().numpy())
# mlp_acc = accuracy_score(y_test.cpu().numpy(), mlp_pred)
# print(f'Scikit-learn MLP Test Accuracy: {mlp_acc:.4f}')