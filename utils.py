import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import copy  # Import the copy module

def train(net, trainloader, criterion, optimizer, device, epochs=5):
    net.to(device)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(labels)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
    print("Finished Training")


def evaluate(net, testloader, device, verbose=True):
    net.to(device)
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    if verbose:
        print(f"Accuracy on test data: {accuracy:.5f}")
    return accuracy


def fold(weights):
    # Poids `weights` est un tensor de dimensions [O, I, K1, K2]
    O, I, K1, K2 = weights.size()
    T = I * K1 * K2
    # On redimensionne les poids pour obtenir une matrice à deux dimensions [O, T]
    folded_weights = weights.view(O, T)
    return folded_weights
def prune_network(model, feature_selection_function, dataset,nbrligne):
    """
    Prune a trained neural network model.

    Parameters:
    model: Trained PyTorch model to prune.
    feature_selection_function: Function to select features from weight tensor.
    dataset: Training dataset used to fine-tune the pruned model.

    Returns:
    A pruned PyTorch model.
    """
    # Copy the model to avoid modifying the original one
    pruned_model = copy.deepcopy(model)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # Get the weight tensor
            weight_tensor = module.weight.data
            
            # If the layer is a convolutional layer, you might need to reshape (fold) the tensor
            if isinstance(module, torch.nn.Conv2d):
                k,l,b,_=weight_tensor.size()
                print(k,l,b)
                (feature_selection_function(k,l,b,nbrligne,module),"ok")
                # break
                # print("-------------------------------")
                weight_tensor = fold(weight_tensor)
               
            # Apply the feature selection function
            # selected_features = feature_selection_function(weight_tensor)
            
            # Update the layer's weights based on selected features
            # This is a placeholder for the update operation
            # module.weight.data = update_weights(weight_tensor, selected_features)
    
    # Prune all unselected connections - this would be done inside the loop, specifics depend on the pruning method
    # Fine-tune the pruned model on the training dataset
    # finetune_model(pruned_model, dataset)
    
    return pruned_model
def feature_selection_function (weight_tensor):
    return weight_tensor

def update_weights(weight_tensor, selected_features):
       return weight_tensor
def  feature_selection_norme ( k,l,b,f,couche): #n k=nombre filter  l=input  f=nombre des min 
 for i in range(k):  # Parcourir les 16 filtres
    for j in range(l):  # Parcourir les 6 canaux de chaque filtre
        norms = torch.norm(couche.weight.data[i, j], p=1, dim=1)  # Calculer la norme L1 pour chaque ligne
        min_norm_indices = torch.topk(norms, f, largest=False).indices  # Trouver les indices des deux lignes avec les plus petites normes L1
        for idx in min_norm_indices:
            couche.weight.data[i, j, idx] = 0  # Mettre à zéro les poids de ces lignes
    
def Variance_(k, l, b, seuil, couche):
    for i in range(k):  # Parcourir les k filtres
        for j in range(l):  # Parcourir les l canaux de chaque filtre
            # Calculer la variance pour chaque ligne du canal j dans le filtre i
            for m in range(b):  # Parcourir les b lignes dans chaque canal
                # Calculez la variance de la ligne m
                variance = torch.var(couche.weight.data[i, j, m, :], unbiased=False)
                # print(variance)
                # Si la variance est inférieure au seuil, mettre à zéro
                if variance < seuil:
                    couche.weight.data[i, j, m, :].fill_(0)

    