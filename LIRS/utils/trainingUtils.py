
import torch
import torch.nn.functional as F
from pynvml import *
import shutil
import glob
from termcolor import colored
from scipy.stats import pearsonr
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from utils.functionalUtils import *
from utils.dataUtils import process_batch
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from torch import optim
from copy import deepcopy
import time
from scipy.stats import entropy
from torchmetrics import AUROC
from matplotlib.backends.backend_pdf import PdfPages
import io
from sklearn.manifold import TSNE
from queue import PriorityQueue
from utils.functionalUtils import *

def calculate_entropy(tensor):
    """
    Calculate the entropy along the C dimension of a tensor of shape (N, K, C).

    Parameters:
    tensor (torch.Tensor): A PyTorch tensor of shape (N, K, C) representing a probability distribution along C.

    Returns:
    torch.Tensor: A tensor of shape (N, K) representing the entropy.
    """
    # Ensure the tensor represents a valid probability distribution along C
    if not torch.all(tensor >= 0) and torch.all(torch.isclose(tensor.sum(dim=-1), torch.tensor(1.0))):
        raise ValueError("Tensor values along C must be non-negative and sum up to 1.")

    # Calculate the entropy
    log_tensor = torch.log(tensor + 1e-9)  # Adding a small value to avoid log(0)
    entropy = -torch.sum(tensor * log_tensor, dim=-1)

    return entropy


def select_top_k_most_certain(graph_emb, unlabeled_train_indices, clf_prob, K=10, C=2):
    """
    Selects the top-K most certain samples for each class based on the predicted probabilities.
    
    Parameters:
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - unlabeled_train_indices: PyTorch Tensor
        The indices of the graphs in the unlabeled training set.
        
    - clf_prob: sklearn CalibratedClassifierCV object
        The trained classifier with probability estimates.
        
    - K: int, optional
        The number of top-K most certain samples to select for each class.
        
    - C: int, optional
        The number of classes.
        
    Returns:
    - List of lists containing the top-K most certain indices for each class.
    """
    
    unlabeled_emb = graph_emb[unlabeled_train_indices].detach().cpu().numpy()
    predicted_proba = clf_prob.predict_proba(unlabeled_emb)
    
    # Compute entropy for each sample
    entropies = np.apply_along_axis(entropy, 1, predicted_proba)
    
    selected_indices = []
    for c in range(C):
        # Filter indices where the maximum predicted probability corresponds to class c
        filtered_indices = np.where(predicted_proba.argmax(axis=1) == c)[0]
        
        # Sort these filtered indices based on their entropies and select the top-K
        sorted_filtered_indices = filtered_indices[np.argsort(entropies[filtered_indices])[:K]]
        
        # Map back to original unlabeled_train_indices
        original_indices = unlabeled_train_indices[torch.tensor(sorted_filtered_indices)].tolist()
        
        selected_indices.append(original_indices)
    print ('len of most certain classes:',len(selected_indices[0]),len(selected_indices[1]))
    return selected_indices


def sample_augmentation(graph_emb, labeled_train_indices,unlabeled_train_indices, top_k_indices, S=10, C=2):
    """
    Selects the top-S closest samples to the centroid for each class.
    
    Parameters:
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - labeled_train_indices: PyTorch Tensor
        The indices of the graphs in the labeled training set.
    
    -unlabeled_train_indices: PyTorch Tensor
        
    - top_k_indices: List of lists
        The top-K most certain indices for each class.
        
    - S: int, optional
        The number of top-S closest samples to select for each class.
        
    - C: int, optional
        The number of classes.
        
    Returns:
    - List of lists containing the top-S closest indices for each class.
    """
    # Calculate the mean vector for each class using labeled_train_indices and top_k_indices
    num_per_class = len(labeled_train_indices) // C
    mean_vectors = []
    
    for c in range(C):
        # class_indices = labeled_train_indices[c * num_per_class : (c + 1) * num_per_class].tolist()
        # class_indices += top_k_indices[c]
        class_indices = top_k_indices[c] #?
         
        class_emb = graph_emb[class_indices].detach().cpu().numpy()
        mean_vector = np.mean(class_emb, axis=0)
        mean_vectors.append(mean_vector)
    
    # Use mean vectors as initial points in K-means clustering over unlabeled_train_indices
    unlabeled_emb = graph_emb[unlabeled_train_indices].detach().cpu().numpy()
    kmeans = KMeans(n_clusters=C, init=np.array(mean_vectors), n_init=10)
    kmeans.fit(unlabeled_emb)
    
    # Get the centroids for each cluster
    centroids = kmeans.cluster_centers_
    centroids = np.array(mean_vectors) #?
    
    # Find the top-S closest samples to each cluster centroid
    selected_indices = []
    for centroid in centroids:
        distances = np.linalg.norm(unlabeled_emb - centroid, axis=1)
        closest_indices = np.argsort(distances)[:S]
        selected_indices.append(unlabeled_train_indices[closest_indices].tolist())
        
    return selected_indices



def update_pseudo_samples(pseudo_samples, train_indices, augmented_sample_indices):
    """
    Update the pseudo_samples list by performing union with augmented_sample_indices
    and removing elements found in train_indices.

    Parameters:
    - pseudo_samples: list
        List of existing pseudo samples.

    - train_indices: PyTorch 1D Tensor
        Tensor containing indices of training samples.

    - augmented_sample_indices: PyTorch 1D Tensor
        Tensor containing indices of augmented samples.

    Returns:
    - updated_pseudo_samples: list
        List of updated pseudo samples.
    """

    # Convert all data to sets for easier manipulation
    pseudo_samples_set = set(pseudo_samples)
    train_indices_set = set(train_indices.cpu().numpy())
    augmented_sample_indices_set = set(augmented_sample_indices.cpu().numpy())

    # Perform union operation to update pseudo_samples
    updated_pseudo_samples_set = pseudo_samples_set.union(augmented_sample_indices_set)

    # Remove elements found in train_indices
    updated_pseudo_samples_set = updated_pseudo_samples_set.difference(train_indices_set)

    # Convert back to list
    updated_pseudo_samples = list(updated_pseudo_samples_set)

    return updated_pseudo_samples



def train_and_evaluate_svm_with_proba(dataset, labeled_train_indices, val_indices, graph_emb, test_indices=None,unlabeled_train_indices=None, C=1.0,cv = 3, K=10, S=10,train_union_indices=None):
    """
    Function to train a Linear SVM on graph embeddings and evaluate its performance.
    
    Parameters:
    - dataset: PyTorch Geometric Dataset
        The dataset containing the graph information and labels.
    
    - labeled_train_indices: PyTorch Tensor
        The indices of the graphs in the training set.
    
    - val_indices: PyTorch Tensor
        The indices of the graphs in the validation set.
        
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - test_indices: PyTorch Tensor, optional
        The indices of the graphs in the test set.
    
    Returns:
    - Validation accuracy of the trained Linear SVM.
    - Probability estimates for the validation set.
    - Test accuracy if test_indices is provided.
    """
    
    # Get the embeddings and labels for the training set
    test_accuracy = None
    train_emb = graph_emb[labeled_train_indices]
    train_labels = dataset.data.y[labeled_train_indices]
    
    # Get the embeddings and labels for the validation set
    val_emb = graph_emb[val_indices]
    val_labels = dataset.data.y[val_indices]
    
    # Convert PyTorch Tensors to NumPy arrays
    train_emb_np = train_emb.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_emb_np = val_emb.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
    # Train a Linear SVM with probability estimates enabled
    clf = LinearSVC(C=C,max_iter=300,tol=0.001)
    clf.fit(train_emb_np, train_labels_np)
    
    # Make predictions on the validation set
    val_predictions = clf.predict(val_emb_np)
    
    # Get probability estimates for the validation set
    clf_prob = CalibratedClassifierCV(base_estimator=clf, method='sigmoid',cv=cv)
    clf_prob.fit(train_emb_np, train_labels_np)
    val_proba = clf_prob.predict_proba(val_emb_np)
    # Compute the validation accuracy
    val_accuracy = accuracy_score(val_labels_np, val_predictions)
    
    if unlabeled_train_indices is not None:
        most_certain_indices = select_top_k_most_certain(graph_emb, unlabeled_train_indices, clf_prob, K=K, C=2)
        sampled_augment_indices = sample_augmentation(graph_emb, labeled_train_indices,unlabeled_train_indices, most_certain_indices, S=S, C=2)
        return val_accuracy, val_proba, sampled_augment_indices,clf_prob    #! watch out for most_certain_indices
    
    if test_indices is not None:
        # Get the embeddings and labels for the test set
        test_emb = graph_emb[test_indices]
        test_labels = dataset.data.y[test_indices]
        
        # Convert to NumPy arrays
        test_emb_np = test_emb.detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        
        # Make predictions on the test set
        test_predictions = clf.predict(test_emb_np)
        
        # Compute the test accuracy
        test_accuracy = accuracy_score(test_labels_np, test_predictions)
    if train_union_indices is not None:
        # Get the embeddings for train_union_indices
        train_union_emb = graph_emb[train_union_indices]
        train_union_emb_np = train_union_emb.detach().cpu().numpy()
        
        # Get prediction probabilities for train_union_emb
        train_union_proba = clf_prob.predict_proba(train_union_emb_np)
        
        # Calculate entropy for the predictions
        entropy = -np.sum(train_union_proba * np.log(train_union_proba + 1e-10), axis=1)
        
        # Get the indices where predictions for class 1 are greater than class 0 and vice versa
        class_1_indices = np.where(train_union_proba[:, 1] > train_union_proba[:, 0])[0]
        class_0_indices = np.where(train_union_proba[:, 0] > train_union_proba[:, 1])[0]

        # Get indices for most certain samples predicted as 1 and 0
        certain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[:50]]
        certain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[:50]]

        # Get indices for most uncertain samples predicted as 1 and 0
        uncertain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[-50:]]
        uncertain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[-50:]]

        # Map back to original indices in train_union_indices and convert to tensors
        certain_1_original_indices = torch.tensor(train_union_indices[certain_1_indices])
        certain_0_original_indices = torch.tensor(train_union_indices[certain_0_indices])
        uncertain_1_original_indices = torch.tensor(train_union_indices[uncertain_1_indices])
        uncertain_0_original_indices = torch.tensor(train_union_indices[uncertain_0_indices])
        return val_accuracy, val_proba, test_accuracy, clf_prob,[certain_1_original_indices, certain_0_original_indices, uncertain_1_original_indices, uncertain_0_original_indices]
    
    return val_accuracy, val_proba, test_accuracy, clf_prob,None
    
def train_and_evaluate_svm_with_proba_synthetic(tr_dataset,val_dataset, labeled_train_indices, tr_graph_emb, val_graph_emb,test_dataset=None ,test_graph_emb=None,unlabeled_train_indices=None, C=1.0,cv = 3, K=30, S=30,train_union_indices=None,metric="acc"):
    """
    Function to train a Linear SVM on graph embeddings and evaluate its performance.
    
    Parameters:
    - dataset: PyTorch Geometric Dataset
        The dataset containing the graph information and labels.
    
    - labeled_train_indices: PyTorch Tensor
        The indices of the graphs in the training set.
    
    - val_indices: PyTorch Tensor
        The indices of the graphs in the validation set.
        
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - test_indices: PyTorch Tensor, optional
        The indices of the graphs in the test set.
    
    Returns:
    - Validation accuracy of the trained Linear SVM.
    - Probability estimates for the validation set.
    - Test accuracy if test_indices is provided.
    """
    
    # Get the embeddings and labels for the training set
    auroc = AUROC(task='binary')
    test_accuracy = None
    train_emb = tr_graph_emb[labeled_train_indices]
    train_labels = tr_dataset.data.y[labeled_train_indices]
    
    # Get the embeddings and labels for the validation set
    val_emb = val_graph_emb
    N = len(val_dataset)
    val_labels = torch.cat([i.y for i in val_dataset]).view(-1,)
    # Convert PyTorch Tensors to NumPy arrays
    train_emb_np = train_emb.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_emb_np = val_emb.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
    # Train a Linear SVM with probability estimates enabled
    clf = LinearSVC(C=C,max_iter=300,tol=0.001)
    clf.fit(train_emb_np, train_labels_np)
    
    # Make predictions on the validation set
    val_predictions = clf.predict(val_emb_np)
    train_predictions = clf.predict(train_emb_np)
    # Get probability estimates for the validation set
    clf_prob = CalibratedClassifierCV(base_estimator=clf, method='sigmoid',cv=cv)
    clf_prob.fit(train_emb_np, train_labels_np)
    val_proba = clf_prob.predict_proba(val_emb_np)
    train_proba = clf_prob.predict_proba(train_emb_np)
    # Compute the validation accuracy
    # print (11111111111111111111111)
    # print (train_emb_np.shape,train_labels.shape)
    # print (val_labels_np.shape)
    # print (val_predictions.shape)
    if metric=='acc':
        val_accuracy = accuracy_score(val_labels_np, val_predictions)
        tr_accuracy = accuracy_score(train_labels_np, train_predictions)
    else:
        val_labels_tensor = torch.tensor(val_labels_np)
        val_predictions_tensor = torch.tensor(val_proba)
        train_predictions_tensor = torch.tensor(train_proba)
        train_labels_tensor = torch.tensor(train_labels_np)
        val_accuracy = auroc(val_predictions_tensor[:,1].view(-1,),val_labels_tensor.view(-1,)).item()
        tr_accuracy = auroc(train_predictions_tensor[:,1].view(-1,),train_labels_tensor.view(-1,)).item()
        
        
    if unlabeled_train_indices is not None:
        most_certain_indices = select_top_k_most_certain(tr_graph_emb, unlabeled_train_indices, clf_prob, K=K, C=2)
        sampled_augment_indices = sample_augmentation(tr_graph_emb, labeled_train_indices,unlabeled_train_indices, most_certain_indices, S=S, C=2)
        return val_accuracy, val_proba, sampled_augment_indices,clf_prob    #! watch out for most_certain_indices
    
    if test_graph_emb is not None:
        # Get the embeddings and labels for the test set
        test_emb = test_graph_emb
        test_labels = test_dataset.data.y
        
        # Convert to NumPy arrays
        test_emb_np = test_emb.detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        
        # Make predictions on the test set
        test_predictions = clf.predict(test_emb_np)
        test_proba = clf_prob.predict_proba(test_emb_np)
        
        # Compute the test accuracy
        if metric=='acc':
            test_accuracy = accuracy_score(test_labels_np, test_predictions)
        else:
            test_labels_tensor = torch.tensor(test_labels_np)
            test_predictions_tensor = torch.tensor(test_proba)
            test_accuracy = auroc(test_predictions_tensor[:,1].view(-1,),test_labels_tensor.view(-1,)).item()
    if train_union_indices is not None:
        # Get the embeddings for train_union_indices
        train_union_emb = tr_graph_emb[train_union_indices]
        train_union_emb_np = train_union_emb.detach().cpu().numpy()

        # Get prediction probabilities for train_union_emb
        train_union_proba = clf_prob.predict_proba(train_union_emb_np)
        
        # Calculate entropy for the predictions
        entropy = -np.sum(train_union_proba * np.log(train_union_proba + 1e-10), axis=1)
        
        # Get the indices where predictions for class 1 are greater than class 0 and vice versa
        class_1_indices = np.where(train_union_proba[:, 1] > train_union_proba[:, 0])[0]
        class_0_indices = np.where(train_union_proba[:, 0] > train_union_proba[:, 1])[0]

        # Get indices for most certain samples predicted as 1 and 0
        certain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[:50]]
        certain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[:50]]

        # Get indices for most uncertain samples predicted as 1 and 0
        uncertain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[-50:]]
        uncertain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[-50:]]

        # Map back to original indices in train_union_indices and convert to tensors
        certain_1_original_indices = torch.tensor(train_union_indices[certain_1_indices])
        certain_0_original_indices = torch.tensor(train_union_indices[certain_0_indices])
        uncertain_1_original_indices = torch.tensor(train_union_indices[uncertain_1_indices])
        uncertain_0_original_indices = torch.tensor(train_union_indices[uncertain_0_indices])
        return val_accuracy, val_proba, test_accuracy, clf_prob,[certain_1_original_indices, certain_0_original_indices, uncertain_1_original_indices, uncertain_0_original_indices]
    
    return val_accuracy, val_proba, test_accuracy, clf_prob,None,tr_accuracy


def train_and_evaluate_svm_with_proba_synthetic_vis(tr_dataset,val_dataset, labeled_train_indices, tr_graph_emb, val_graph_emb,test_dataset=None ,test_graph_emb=None,unlabeled_train_indices=None, C=1.0,cv = 3, K=30, S=30,train_union_indices=None,metric="acc"):
    """
    Function to train a Linear SVM on graph embeddings and evaluate its performance.
    
    Parameters:
    - dataset: PyTorch Geometric Dataset
        The dataset containing the graph information and labels.
    
    - labeled_train_indices: PyTorch Tensor
        The indices of the graphs in the training set.
    
    - val_indices: PyTorch Tensor
        The indices of the graphs in the validation set.
        
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - test_indices: PyTorch Tensor, optional
        The indices of the graphs in the test set.
    
    Returns:
    - Validation accuracy of the trained Linear SVM.
    - Probability estimates for the validation set.
    - Test accuracy if test_indices is provided.
    """
    
    # Get the embeddings and labels for the training set
    auroc = AUROC(task='binary')
    test_accuracy = None
    train_emb = tr_graph_emb[labeled_train_indices]
    train_labels = tr_dataset.data.y[labeled_train_indices]
    train_corr = tr_dataset.data.high_corr[labeled_train_indices]
    # Get the embeddings and labels for the validation set
    val_emb = val_graph_emb
    N = len(val_dataset)
    val_labels = val_dataset.data.y
    # Convert PyTorch Tensors to NumPy arrays
    train_emb_np = train_emb.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    train_corr_np = train_corr.detach().cpu().numpy()
    val_emb_np = val_emb.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
    # Train a Linear SVM with probability estimates enabled
    
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    train_emb_2d = tsne.fit_transform(train_emb_np)
    # Train a Linear SVM with the 2D embeddings
    clf = LinearSVC(C=C, max_iter=300, tol=0.001)
    clf.fit(train_emb_2d, train_labels_np)
    # Plotting decision boundary
    x_min, x_max = train_emb_2d[:, 0].min() - 1, train_emb_2d[:, 0].max() + 1
    y_min, y_max = train_emb_2d[:, 1].min() - 1, train_emb_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Create a PDF object
    pdf_output = io.BytesIO()
    with PdfPages(pdf_output) as pdf:
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(train_emb_2d[:, 0], train_emb_2d[:, 1], c=train_corr_np, cmap=plt.cm.RdBk, s=20, edgecolor='k')
        plt.title("Decision Boundary with t-SNE Reduced Data")
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    
    # Make predictions on the validation set
    train_predictions = clf.predict(train_emb_2d)
    # Get probability estimates for the validation set
    clf_prob = CalibratedClassifierCV(base_estimator=clf, method='sigmoid',cv=cv)
    clf_prob.fit(train_emb_2d, train_labels_np)
    if metric=='acc':
        tr_accuracy = accuracy_score(train_labels_np, train_predictions)
    
    return -1., -1., -1., clf_prob,None,tr_accuracy,pdf_output

def train_and_evaluate_mlp_synthetic(tr_dataset,val_dataset, labeled_train_indices, tr_graph_emb, val_graph_emb,test_dataset=None,test_graph_emb=None,unlabeled_train_indices=None, K=30, S=30,train_union_indices=None):
    """
    Function to train a MLP on graph embeddings and evaluate its performance.
    
    Parameters:
    - dataset: PyTorch Geometric Dataset
        The dataset containing the graph information and labels.
    
    - labeled_train_indices: PyTorch Tensor
        The indices of the graphs in the training set.
    
    - val_indices: PyTorch Tensor
        The indices of the graphs in the validation set.
        
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - test_indices: PyTorch Tensor, optional
        The indices of the graphs in the test set.
    
    Returns:
    - Validation accuracy of the trained Linear SVM.
    - Probability estimates for the validation set.
    - Test accuracy if test_indices is provided.
    """
    
    # Get the embeddings and labels for the training set
    test_accuracy = None
    train_emb = tr_graph_emb[labeled_train_indices]
    train_labels = tr_dataset.data.y[labeled_train_indices]
    
    # Get the embeddings and labels for the validation set
    val_emb = val_graph_emb
    val_labels = val_dataset.data.y
    
    # Convert PyTorch Tensors to NumPy arrays
    train_emb_np = train_emb.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_emb_np = val_emb.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
    # Train a Linear SVM with probability estimates enabled
    clf = MLPClassifier(hidden_layer_sizes=(64,32,32),max_iter=200,tol=0.001,activation='relu')
    clf.fit(train_emb_np, train_labels_np)
    
    # Make predictions on the validation set
    val_predictions = clf.predict(val_emb_np)
    train_predictions = clf.predict(train_emb_np)
    # Get probability estimates for the validation set
    val_proba = clf.predict_proba(val_emb_np)
    # Compute the validation accuracy
    val_accuracy = accuracy_score(val_labels_np, val_predictions)
    tr_accuracy = accuracy_score(train_labels_np, train_predictions)
    
    if unlabeled_train_indices is not None:
        most_certain_indices = select_top_k_most_certain(tr_graph_emb, unlabeled_train_indices, clf, K=K, C=2)
        sampled_augment_indices = sample_augmentation(tr_graph_emb, labeled_train_indices,unlabeled_train_indices, most_certain_indices, S=S, C=2)
        return val_accuracy, val_proba, sampled_augment_indices,clf    #! watch out for most_certain_indices
    
    if test_graph_emb is not None:
        # Get the embeddings and labels for the test set
        test_emb = test_graph_emb
        test_labels = test_dataset.data.y
        
        # Convert to NumPy arrays
        test_emb_np = test_emb.detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        
        # Make predictions on the test set
        test_predictions = clf.predict(test_emb_np)
        
        # Compute the test accuracy
        test_accuracy = accuracy_score(test_labels_np, test_predictions)
    if train_union_indices is not None:
        # Get the embeddings for train_union_indices
        train_union_emb = tr_graph_emb[train_union_indices]
        train_union_emb_np = train_union_emb.detach().cpu().numpy()
        
        # Get prediction probabilities for train_union_emb
        train_union_proba = clf.predict_proba(train_union_emb_np)
        
        # Calculate entropy for the predictions
        entropy = -np.sum(train_union_proba * np.log(train_union_proba + 1e-10), axis=1)
        
        # Get the indices where predictions for class 1 are greater than class 0 and vice versa
        class_1_indices = np.where(train_union_proba[:, 1] > train_union_proba[:, 0])[0]
        class_0_indices = np.where(train_union_proba[:, 0] > train_union_proba[:, 1])[0]

        # Get indices for most certain samples predicted as 1 and 0
        certain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[:50]]
        certain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[:50]]

        # Get indices for most uncertain samples predicted as 1 and 0
        uncertain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[-50:]]
        uncertain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[-50:]]

        # Map back to original indices in train_union_indices and convert to tensors
        certain_1_original_indices = torch.tensor(train_union_indices[certain_1_indices])
        certain_0_original_indices = torch.tensor(train_union_indices[certain_0_indices])
        uncertain_1_original_indices = torch.tensor(train_union_indices[uncertain_1_indices])
        uncertain_0_original_indices = torch.tensor(train_union_indices[uncertain_0_indices])
        return val_accuracy, val_proba, test_accuracy, clf,[certain_1_original_indices, certain_0_original_indices, uncertain_1_original_indices, uncertain_0_original_indices]
    
    return val_accuracy, val_proba, test_accuracy, clf,None,tr_accuracy


def train_and_evaluate_svm(X, y, K, S,num_trials = 10):
    """
    Train a linear SVM model on a subset of the data, augment the training set with top-S confident samples,
    and evaluate its accuracy on a validation set.
    
    Parameters:
    - X (numpy.ndarray): Feature matrix of shape (N, D).
    - y (numpy.ndarray): Label vector of shape (N,).
    - K (int): Number of samples to select from each class (0 and 1) for initial training.
    - S (int): Number of top-confident samples to select from each class for augmentation.
    
    Returns:
    - object: Trained SVM model.
    - float: Accuracy of the trained model on the validation set.
    - list: Indices of the initially selected samples from class 0.
    - list: Indices of the initially selected samples from class 1.
    - list: Indices of the augmented selected samples from class 0.
    - list: Indices of the augmented selected samples from class 1.
    """
    
    kmeans_acc_result_without_svm = []
    kmeans_acc_result_with_svm = []
    svm_result = []
    
    for _ in tqdm(range(num_trials)):
        # Step 1: Split the data into training and validation sets (80%:20%)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # Step 2: Randomly select K samples from each class (0 and 1) from the training set
        indices_0 = np.where(y_train == 0)[0]
        indices_1 = np.where(y_train == 1)[0]
        
        selected_indices_0 = np.random.choice(indices_0, K, replace=False)
        selected_indices_1 = np.random.choice(indices_1, K, replace=False)
        
        # Remove selected indices from the pool
        remaining_indices_0 = np.setdiff1d(indices_0, selected_indices_0)
        remaining_indices_1 = np.setdiff1d(indices_1, selected_indices_1)
        
        # Step 3: Train a linear SVM model with probability estimates
        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(X_train[selected_indices_0.tolist() + selected_indices_1.tolist()], y_train[selected_indices_0.tolist() + selected_indices_1.tolist()])
        
        # Step 4: Evaluate the model on the validation set
        y_pred = svm_model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        
        # Step 5: Predict label probabilities on X_train
        y_prob_train = svm_model.predict_proba(X_train)[:, 1]
        
        # Step 6: Get the top-S confident samples for each class and their indices
        sorted_indices_0 = np.argsort(y_prob_train[remaining_indices_0])
        sorted_indices_1 = np.argsort(y_prob_train[remaining_indices_1])[::-1]
        
        top_S_indices_0 = remaining_indices_0[sorted_indices_0[:S]]
        top_S_indices_1 = remaining_indices_1[sorted_indices_1[:S]]
        
        # Step 7: Augment the selected_indices with top-S indices
        augmented_selected_indices_0 = np.concatenate([selected_indices_0, top_S_indices_0])
        augmented_selected_indices_1 = np.concatenate([selected_indices_1, top_S_indices_1])
        
        
        svm_result.append(accuracy)
        
        #! without svm prediction augmentation
        centroids = run_kmeans_with_initial_points(X_train, y_train,indices_0=selected_indices_0,indices_1=selected_indices_1)
        l = find_closest_points(X_train, y_train,centroids,5)
        acc5 = calculate_accuracy(l[0])
        l = find_closest_points(X_train, y_train,centroids,10)
        acc10 = calculate_accuracy(l[0])
        kmeans_acc_result_without_svm.append([acc5,acc10])
        
        
        #! with svm prediction augmentation
        centroids = run_kmeans_with_initial_points(X_train, y_train,indices_0=augmented_selected_indices_0,indices_1=augmented_selected_indices_1)
        l = find_closest_points(X_train, y_train,centroids,5)
        acc5 = calculate_accuracy(l[0])
        l = find_closest_points(X_train, y_train,centroids,10)
        acc10 = calculate_accuracy(l[0])
        kmeans_acc_result_with_svm.append([acc5,acc10])        
        
            
    kmeans_acc_result_without_svm =  np.asarray(kmeans_acc_result_without_svm)
    kmeans_acc_result_without_svm_acc = np.mean(kmeans_acc_result_without_svm,axis=0)
    kmeans_acc_result_without_svm_std = np.std(kmeans_acc_result_without_svm,axis=0)

    kmeans_acc_result_with_svm =  np.asarray(kmeans_acc_result_with_svm)
    kmeans_acc_result_with_svm_acc = np.mean(kmeans_acc_result_with_svm,axis=0)
    kmeans_acc_result_with_svm_std = np.std(kmeans_acc_result_with_svm,axis=0)
    
    
    svm_result = np.asarray(svm_result)
    svm_result_mean = svm_result.mean()
    svm_result_std = svm_result.std()
    msg = f"svm result: {svm_result_mean} +- {svm_result_std}\n kmeans result without svm (Top-5 & Top-10): {kmeans_acc_result_without_svm_acc} +- {kmeans_acc_result_without_svm_std}\n kmeans result with svm (Top-5 & Top-10): {kmeans_acc_result_with_svm_acc} +- {kmeans_acc_result_with_svm_std}"
    return msg


def train_and_evaluate_svm_with_proba_ogbg(tr_dataset,val_dataset, labeled_train_indices, tr_graph_emb, val_graph_emb,test_dataset=None, test_graph_emb=None,unlabeled_train_indices=None, C=1.0,cv = 3, K=30, S=30,train_union_indices=None):
    """
    Function to train a Linear SVM on graph embeddings and evaluate its performance.
    
    Parameters:
    - dataset: PyTorch Geometric Dataset
        The dataset containing the graph information and labels.
    
    - labeled_train_indices: PyTorch Tensor
        The indices of the graphs in the training set.
    
    - val_indices: PyTorch Tensor
        The indices of the graphs in the validation set.
        
    - graph_emb: PyTorch Tensor
        The embeddings for all graphs in the dataset.
        
    - test_indices: PyTorch Tensor, optional
        The indices of the graphs in the test set.
    
    Returns:
    - Validation accuracy of the trained Linear SVM.
    - Probability estimates for the validation set.
    - Test accuracy if test_indices is provided.
    """
    
    # Get the embeddings and labels for the training set
    test_accuracy = None
    train_emb = tr_graph_emb[labeled_train_indices]
    train_labels = []
    for idx in labeled_train_indices:
        train_labels.append(tr_dataset[idx].y.item())
    train_labels = torch.tensor(train_labels)
    # Get the embeddings and labels for the validation set
    val_emb = val_graph_emb
    N = len(val_dataset)
    val_labels = torch.tensor([val_dataset[i].y.item() for i in range(N)])
    # Convert PyTorch Tensors to NumPy arrays
    train_emb_np = train_emb.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_emb_np = val_emb.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
    # Train a Linear SVM with probability estimates enabled
    clf = LinearSVC(C=C,max_iter=300,tol=0.001)
    clf.fit(train_emb_np, train_labels_np)
    
    # Make predictions on the validation set
    val_predictions = clf.predict(val_emb_np)
    train_predictions = clf.predict(train_emb_np)
    # Get probability estimates for the validation set
    clf_prob = CalibratedClassifierCV(base_estimator=clf, method='sigmoid',cv=cv)
    clf_prob.fit(train_emb_np, train_labels_np)
    val_proba = clf_prob.predict_proba(val_emb_np)
    # Compute the validation accuracy
    val_accuracy = accuracy_score(val_labels_np, val_predictions)
    tr_accuracy = accuracy_score(train_labels_np, train_predictions)
    val_auc = roc_auc_score(val_labels_np, val_proba[:,1])
    tr_auc = roc_auc_score(train_labels_np, clf_prob.predict_proba(train_emb_np)[:,1])
    
    if test_graph_emb is not None:
        # Get the embeddings and labels for the test set
        test_emb = test_graph_emb
        N = len(test_dataset)
        test_labels = torch.tensor([test_dataset[i].y.item() for i in range(N)])
        
        # Convert to NumPy arrays
        test_emb_np = test_emb.detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        
        # Make predictions on the test set
        test_predictions = clf.predict(test_emb_np)
        
        # Compute the test accuracy
        test_accuracy = accuracy_score(test_labels_np, test_predictions)
        test_auc = roc_auc_score(test_labels_np, clf_prob.predict_proba(test_emb_np)[:,1])
    if train_union_indices is not None:
        # Get the embeddings for train_union_indices
        train_union_emb = tr_graph_emb[train_union_indices]
        train_union_emb_np = train_union_emb.detach().cpu().numpy()
        
        # Get prediction probabilities for train_union_emb
        train_union_proba = clf_prob.predict_proba(train_union_emb_np)
        
        # Calculate entropy for the predictions
        entropy = -np.sum(train_union_proba * np.log(train_union_proba + 1e-10), axis=1)
        
        # Get the indices where predictions for class 1 are greater than class 0 and vice versa
        class_1_indices = np.where(train_union_proba[:, 1] > train_union_proba[:, 0])[0]
        class_0_indices = np.where(train_union_proba[:, 0] > train_union_proba[:, 1])[0]

        # Get indices for most certain samples predicted as 1 and 0
        certain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[:50]]
        certain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[:50]]

        # Get indices for most uncertain samples predicted as 1 and 0
        uncertain_1_indices = class_1_indices[np.argsort(entropy[class_1_indices])[-50:]]
        uncertain_0_indices = class_0_indices[np.argsort(entropy[class_0_indices])[-50:]]

        # Map back to original indices in train_union_indices and convert to tensors
        certain_1_original_indices = torch.tensor(train_union_indices[certain_1_indices])
        certain_0_original_indices = torch.tensor(train_union_indices[certain_0_indices])
        uncertain_1_original_indices = torch.tensor(train_union_indices[uncertain_1_indices])
        uncertain_0_original_indices = torch.tensor(train_union_indices[uncertain_0_indices])
        return val_accuracy, val_proba, test_accuracy, clf_prob,[certain_1_original_indices, certain_0_original_indices, uncertain_1_original_indices, uncertain_0_original_indices]
    
    return val_auc, val_proba, test_auc, clf_prob,None,tr_auc


def train_model_new_loss(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',detectLabelSmoothing=False,block_mask=False,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,SL_reg = 1.0,model_type='GA-MLP',reweighting_dict=None,metric='acc',patience=10,track_loss= False,data_sampling=False):
    
    if track_loss:
        loss_low_corr = []
        loss_high_corr = []
        loss_dis = []
        loss_low_corr_per_epoch = []
        loss_high_corr_per_epoch = []
        loss_dis_per_epoch = []
    
    
    auroc = AUROC(task='binary')
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=patience, min_lr=1e-3)

    epochs_since_improvement = 0
    best_valid_acc = -1.0  # Initialize best validation accuracy
    test_acc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        loss_low_corr_per_epoch = []
        loss_high_corr_per_epoch = []
        loss_dis_per_epoch = []
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for batch in train_dloader:
            if data_sampling:
                batch = process_batch(batch,num_classes,32)
            optimizer.zero_grad()
            if 0:
                output = model(batch.to(device),mask=True)
            else:
                # output = model(batch.to(device))
                output = model(batch.to(device))
            if not useKD:
                if not detectLabelSmoothing:
                    loss = F.cross_entropy(output,batch.y)
                else:
                    # Check if batch.is_noisy is all False
                    if torch.all(batch.is_noisy == 0):
                        loss = F.cross_entropy(output, batch.y)
                        loss.backward()
                        optimizer.step()
                    else:
                        # Split the output based on batch.is_noisy
                        output_noisy = output[batch.is_noisy == 1]
                        output_not_noisy = output[batch.is_noisy == 0]
                        # Compute the loss for each part
                        loss_noisy = F.cross_entropy(output_noisy, batch.y[batch.is_noisy == True],label_smoothing=0.2)
                        loss_not_noisy = F.cross_entropy(output_not_noisy, batch.y[batch.is_noisy == False])
                        # Combine the losses
                        loss = (loss_noisy * output_noisy.size(0) + loss_not_noisy * output_not_noisy.size(0)) / output.size(0)
                        loss.backward()
                        optimizer.step()
            else:
                # Handle is_noisy attribute
                svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)
                loss_noisy = torch.tensor(0.).to(device)
                loss_not_noisy = torch.tensor(0.).to(device)
                mask_noisy = batch.is_noisy == 1
                mask_not_noisy = batch.is_noisy == 0

                output_noisy = output[mask_noisy]   # all false
                output_not_noisy = output[mask_not_noisy]
                
                y_noisy = batch.y[mask_noisy]
                y_not_noisy = batch.y[mask_not_noisy]
                
                # Regular cross-entropy for not noisy data
                if torch.any(mask_not_noisy):
                    loss_not_noisy = F.cross_entropy(output_not_noisy, y_not_noisy)
                
                if track_loss:
                    mask_high_corr = (batch.is_noisy == 0) & (batch.high_corr == 1)
                    mask_low_corr= (batch.is_noisy == 0) & (batch.high_corr == 0)
                    y_high_corr = batch.y[mask_high_corr]
                    y_low_corr = batch.y[mask_low_corr]
                    output_high_corr = output[mask_high_corr]
                    output_low_corr = output[mask_low_corr]
                    
                    loss_high_corr_per_epoch.append(F.cross_entropy(output_high_corr, y_high_corr,reduction='none'))
                    loss_low_corr_per_epoch.append(F.cross_entropy(output_low_corr, y_low_corr,reduction='none'))
                    
                # Handle svm_proba attribute
                
                svm_proba = batch.svm_proba  
                svm_proba = svm_proba.view(-1,5,num_classes) # shape (N, 5, C)
                output_prob = F.log_softmax(output, dim=1) # shape (N,  C)
                N = output_prob.shape[0]
                svm_proba = torch.tensor([[1./3,1./3,1./3]]).to('cuda')
                loss_svm = torch.sum(-svm_proba * output_prob)/N  # Shape (N, 5). kl div between model predicts and svm predicts
                
                # loss_svm = torch.mean(loss_svm[:,0])  #! without weighted loss
                # # Final loss as the average of the 5 losses
                # loss_svm_avg = torch.mean(loss_svm)
                # Combine all losses
                loss = loss_not_noisy + SL_reg*loss_svm
                loss.backward()
                optimizer.step()
                loss_dis_per_epoch.append(loss_svm.item())
        
        if track_loss:
            loss_high_corr_per_epoch = torch.cat(loss_high_corr_per_epoch,dim=0)
            loss_low_corr_per_epoch = torch.cat(loss_low_corr_per_epoch,dim=0)
            high_mask = ~torch.isnan(loss_high_corr_per_epoch)
            low_mask = ~torch.isnan(loss_low_corr_per_epoch)
            high_count = torch.sum(high_mask)
            low_count = torch.sum(low_mask)
            loss_high_corr_per_epoch_mean = torch.sum(loss_high_corr_per_epoch[high_mask])/high_count
            loss_low_corr_per_epoch_mean = torch.sum(loss_low_corr_per_epoch[low_mask])/low_count
            loss_high_corr.append(loss_high_corr_per_epoch_mean.item())
            loss_low_corr.append(loss_low_corr_per_epoch_mean.item())
            loss_dis.append(np.mean(np.asarray(loss_dis_per_epoch)))
        
        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for batch in train_dloader:
                output = model(batch.to(device))
                # Mask for non-noisy data
                if "is_noisy" in batch:
                    mask_not_noisy = batch.is_noisy == 0
                    # Select only non-noisy predictions and labels
                    train_preds.append(torch.argmax(output, dim=1)[mask_not_noisy].cpu().numpy())
                    train_labels.append(batch.y[mask_not_noisy].cpu().numpy())
                    train_probs.append(F.softmax(output,dim=1)[mask_not_noisy].cpu().numpy())
                else:
                    train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                    train_labels.append(batch.y.cpu().numpy())  
                    train_probs.append(F.softmax(output,dim=1).cpu().numpy())
            
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        if metric=='acc':
            train_acc = accuracy_score(train_labels, train_preds)
        else:
            train_labels_tensor = torch.tensor(train_labels).view(-1,)
            train_preds_tensor = torch.tensor(train_probs[:,1]).view(-1,)
            train_acc = auroc(train_preds_tensor,train_labels_tensor).item()
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for batch in valid_dloader:
                output = model(batch.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        if metric=='acc':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_labels_tensor = torch.tensor(val_labels).view(-1,)
            val_preds_tensor = torch.tensor(val_pred_proba_epoch[:,1]).view(-1,)
            val_acc = auroc(val_preds_tensor,val_labels_tensor).item()
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_acc)
        
        if val_acc > best_valid_acc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = val_acc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc = auroc(test_preds_tensor,test_labels_tensor).item()
                    test_acc_tmp = test_acc
            best_model = deepcopy(model)  # Save the best model
        else:
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc_tmp = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc_tmp = auroc(test_preds_tensor,test_labels_tensor).item()
        print (colored(f'epoch:{epoch},train acc:{train_acc}, val_acc:{val_acc}, test_acc:{test_acc_tmp}','red','on_white'))
        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
    
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'metric_name:{metric}, {model_type}: best_train_acc:{best_train_acc}, best valid_acc:{best_valid_acc}, test_acc:{test_acc}','red','on_yellow'))
    if track_loss:
        return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba,(loss_high_corr,loss_low_corr,loss_dis)
    else:
        return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba


def train_model_SSL(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',detectLabelSmoothing=False,block_mask=False,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,SL_reg = 1.0,model_type='GA-MLP',reweighting_dict=None,metric='acc',patience=10,track_loss= False,data_sampling=False):
    
    auroc = AUROC(task='binary')
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=patience, min_lr=1e-3)

    epochs_since_improvement = 0
    best_valid_acc = -1.0  # Initialize best validation accuracy
    test_acc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        loss_dis_per_epoch = []
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for idx,emb, y,svm_proba in train_dloader:
            optimizer.zero_grad()
            emb, y = emb.to(device), y.to(device)
            output = model(emb) 
            svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)
            loss_not_noisy = torch.tensor(0.).to(device)
            
            loss_not_noisy = F.cross_entropy(output, y)
            
            svm_proba = svm_proba.view(-1,5,num_classes).to(device) # shape (N, 5, C)
            output_prob = F.log_softmax(output, dim=1).unsqueeze(1) # shape (N, 1, C)
            loss_svm = torch.sum(-svm_proba * output_prob, dim=2)  # Shape (N, 5). kl div between model predicts and svm predicts
            
            if reweighting_dict is not None and epoch>=5:
                # Assuming batch.id is a tensor of shape (N,)
                reweight_factors_list = [reweighting_dict[i.item()] for i in idx]

                # Convert the list of lists to a 2D tensor of shape (N, 5)
                reweight_factors = torch.tensor(reweight_factors_list, dtype=torch.float32).to(device)
                # Apply the reweighting to the loss
                loss_svm = loss_svm * reweight_factors       
                
                #! adaptive KD
                weighted_acc = reweight_valid_acc(svm_valid_accs,temperature=temp).to(device)
                loss_svm = torch.mean(torch.sum(weighted_acc*loss_svm,dim=1)) # KD loss
                # loss_svm = torch.mean(loss_svm[:,0])  #! without weighted loss
                # # Final loss as the average of the 5 losses
                # loss_svm_avg = torch.mean(loss_svm)
                # Combine all losses
                loss = loss_not_noisy + SL_reg*loss_svm
                loss.backward()
                optimizer.step()
                loss_dis_per_epoch.append(loss_svm.item())
        
        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for _,emb, y,_ in train_dloader:
                output = model(emb.to(device))
                # Mask for non-noisy data
                train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                train_labels.append(y.cpu().numpy())  
                train_probs.append(F.softmax(output,dim=1).cpu().numpy())
            
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        if metric=='acc':
            train_acc = accuracy_score(train_labels, train_preds)
        else:
            train_labels_tensor = torch.tensor(train_labels).view(-1,)
            train_preds_tensor = torch.tensor(train_probs[:,1]).view(-1,)
            train_acc = auroc(train_preds_tensor,train_labels_tensor).item()
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for emb,y in valid_dloader:
                output = model(emb.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        if metric=='acc':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_labels_tensor = torch.tensor(val_labels).view(-1,)
            val_preds_tensor = torch.tensor(val_pred_proba_epoch[:,1]).view(-1,)
            val_acc = auroc(val_preds_tensor,val_labels_tensor).item()
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_acc)
        
        if val_acc > best_valid_acc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = val_acc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for emb,y in test_dloader:
                        output = model(emb.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc = auroc(test_preds_tensor,test_labels_tensor).item()
                    test_acc_tmp = test_acc
            best_model = deepcopy(model)  # Save the best model
        else:
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for emb,y in test_dloader:
                        output = model(emb.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc_tmp = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc_tmp = auroc(test_preds_tensor,test_labels_tensor).item()
        print (colored(f'epoch:{epoch},train acc:{train_acc}, val_acc:{val_acc}, test_acc:{test_acc_tmp}','red','on_white'))
        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
    
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'metric_name:{metric}, {model_type}: best_train_acc:{best_train_acc}, best valid_acc:{best_valid_acc}, test_acc:{test_acc}','red','on_yellow'))
    return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba


def train_model(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',detectLabelSmoothing=False,block_mask=False,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,sample_entropy_temp=1.0,SL_reg = 1.0,model_type='GA-MLP',reweighting_dict=None,metric='acc',patience=10,track_loss= False,data_sampling=False,q=0.5):
    
    if track_loss:
        loss_low_corr = []
        loss_high_corr = []
        loss_dis = []
        loss_low_corr_per_epoch = []
        loss_high_corr_per_epoch = []
        loss_dis_per_epoch = []
    
    auroc = AUROC(task='binary')
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=patience, min_lr=1e-3)

    epochs_since_improvement = 0
    best_valid_acc = -1.0  # Initialize best validation accuracy
    test_acc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        loss_low_corr_per_epoch = []
        loss_high_corr_per_epoch = []
        loss_dis_per_epoch = []
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for batch in train_dloader:
            
            optimizer.zero_grad()
            output = model(batch.to(device))
            # Handle is_noisy attribute
            svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)
            y = batch.y
            env_logits = batch.logits
            if track_loss:
                mask_high_corr = batch.high_corr == 1
                mask_low_corr= batch.high_corr == 0
                y_high_corr = batch.y[mask_high_corr]
                y_low_corr = batch.y[mask_low_corr]
                output_high_corr = output[mask_high_corr]
                output_low_corr = output[mask_low_corr]
                
                loss_high_corr_per_epoch.append(F.cross_entropy(output_high_corr, y_high_corr,reduction='none'))
                loss_low_corr_per_epoch.append(F.cross_entropy(output_low_corr, y_low_corr,reduction='none'))
            
            # Handle svm_proba attribute
            
            y_hat = F.cross_entropy(output, y,reduction='mean')
            
            # calc entropy
            if sample_entropy_temp>0:
                env_entropy = calculate_entropy(env_logits)
                env_entropy_weight = softmax_with_temperature_v2(-1.*env_entropy,sample_entropy_temp)  # (N,K) for K svms
            else:
                env_entropy_weight = 1.0
            
            loss_components = []
            for i in range(env_logits.shape[1]):
                loss_svm = F.cross_entropy(output, env_logits[:,i,:],reduction='none')
                w_ = calculate_sample_weights(env_logits[:,i,:],y,class_num=3,q=q,device=device)
                s_ = torch.mean(loss_svm*w_)
                loss_components.append(s_)
            
            loss_components = torch.cat([x.unsqueeze(0) for x in loss_components]).view(-1,)
            #! adaptive KD
            weighted_acc = reweight_valid_acc(svm_valid_accs,temperature=temp).to(device)
            loss_val = y_hat + SL_reg*torch.sum(weighted_acc*loss_components)
            loss_val.backward()
            optimizer.step()
            loss_dis_per_epoch.append(loss_val.item())
        
        if track_loss:
            loss_high_corr_per_epoch = torch.cat(loss_high_corr_per_epoch,dim=0)
            loss_low_corr_per_epoch = torch.cat(loss_low_corr_per_epoch,dim=0)
            high_mask = ~torch.isnan(loss_high_corr_per_epoch)
            low_mask = ~torch.isnan(loss_low_corr_per_epoch)
            high_count = torch.sum(high_mask)
            low_count = torch.sum(low_mask)
            loss_high_corr_per_epoch_mean = torch.sum(loss_high_corr_per_epoch[high_mask])/high_count
            loss_low_corr_per_epoch_mean = torch.sum(loss_low_corr_per_epoch[low_mask])/low_count
            loss_high_corr.append(loss_high_corr_per_epoch_mean.item())
            loss_low_corr.append(loss_low_corr_per_epoch_mean.item())
            loss_dis.append(np.mean(np.asarray(loss_dis_per_epoch)))
        
        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for batch in train_dloader:
                output = model(batch.to(device))
                # Mask for non-noisy data
                train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                train_labels.append(batch.y.cpu().numpy())  
                train_probs.append(F.softmax(output,dim=1).cpu().numpy())
        
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        if metric=='acc':
            train_acc = accuracy_score(train_labels, train_preds)
        else:
            train_labels_tensor = torch.tensor(train_labels).view(-1,)
            train_preds_tensor = torch.tensor(train_probs[:,1]).view(-1,)
            train_acc = auroc(train_preds_tensor,train_labels_tensor).item()
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for batch in valid_dloader:
                output = model(batch.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        if metric=='acc':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_labels_tensor = torch.tensor(val_labels).view(-1,)
            val_preds_tensor = torch.tensor(val_pred_proba_epoch[:,1]).view(-1,)
            val_acc = auroc(val_preds_tensor,val_labels_tensor).item()
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_acc)
        
        if val_acc > best_valid_acc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = val_acc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc = auroc(test_preds_tensor,test_labels_tensor).item()
                    test_acc_tmp = test_acc
            best_model = deepcopy(model)  # Save the best model
        else:
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc_tmp = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc_tmp = auroc(test_preds_tensor,test_labels_tensor).item()
        print (colored(f'epoch:{epoch},train acc:{train_acc}, val_acc:{val_acc}, test_acc:{test_acc_tmp}','red','on_white'))
        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
    
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'metric_name:{metric}, {model_type}: best_train_acc:{best_train_acc}, best valid_acc:{best_valid_acc}, test_acc:{test_acc}','red','on_yellow'))
    if track_loss:
        return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba,(loss_high_corr,loss_low_corr,loss_dis)
    else:
        return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba




def train_model_without_distillation(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',detectLabelSmoothing=False,block_mask=False,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,sample_entropy_temp=1.0,SL_reg = 1.0,model_type='GA-MLP',reweighting_dict=None,metric='acc',patience=10,track_loss= False,data_sampling=False,q=0.5):
    
    if track_loss:
        loss_low_corr = []
        loss_high_corr = []
        loss_dis = []
        loss_low_corr_per_epoch = []
        loss_high_corr_per_epoch = []
        loss_dis_per_epoch = []
    
    auroc = AUROC(task='binary')
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=patience, min_lr=1e-3)

    epochs_since_improvement = 0
    best_valid_acc = -1.0  # Initialize best validation accuracy
    test_acc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        loss_low_corr_per_epoch = []
        loss_high_corr_per_epoch = []
        loss_dis_per_epoch = []
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for batch in train_dloader:
            
            optimizer.zero_grad()
            output = model(batch.to(device))
            # Handle is_noisy attribute
            svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)
            y = batch.y
            env_logits = batch.logits
            if track_loss:
                mask_high_corr = batch.high_corr == 1
                mask_low_corr= batch.high_corr == 0
                y_high_corr = batch.y[mask_high_corr]
                y_low_corr = batch.y[mask_low_corr]
                output_high_corr = output[mask_high_corr]
                output_low_corr = output[mask_low_corr]
                loss_high_corr_per_epoch.append(F.cross_entropy(output_high_corr, y_high_corr,reduction='none'))
                loss_low_corr_per_epoch.append(F.cross_entropy(output_low_corr, y_low_corr,reduction='none'))
            
            # Handle svm_proba attribute
            y_hat = F.cross_entropy(output, y,reduction='none')
            loss_components = []
            for i in range(env_logits.shape[1]):
                loss_svm = F.cross_entropy(output, env_logits[:,i,:],reduction='none')
                w_ = calculate_sample_weights(env_logits[:,i,:],y,class_num=3,q=q,device=device)
                s_ = torch.mean(y_hat*w_)
                loss_components.append(s_)
            
            loss_components = torch.cat([x.unsqueeze(0) for x in loss_components]).view(-1,)
            #! adaptive KD
            weighted_acc = reweight_valid_acc(svm_valid_accs,temperature=temp).to(device)
            loss_val = torch.sum(weighted_acc*loss_components)
            loss_val.backward()
            optimizer.step()
        
        if track_loss:
            loss_high_corr_per_epoch = torch.cat(loss_high_corr_per_epoch,dim=0)
            loss_low_corr_per_epoch = torch.cat(loss_low_corr_per_epoch,dim=0)
            high_mask = ~torch.isnan(loss_high_corr_per_epoch)
            low_mask = ~torch.isnan(loss_low_corr_per_epoch)
            high_count = torch.sum(high_mask)
            low_count = torch.sum(low_mask)
            loss_high_corr_per_epoch_mean = torch.sum(loss_high_corr_per_epoch[high_mask])/high_count
            loss_low_corr_per_epoch_mean = torch.sum(loss_low_corr_per_epoch[low_mask])/low_count
            loss_high_corr.append(loss_high_corr_per_epoch_mean.item())
            loss_low_corr.append(loss_low_corr_per_epoch_mean.item())
            loss_dis.append(np.mean(np.asarray(loss_dis_per_epoch)))
        
        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for batch in train_dloader:
                output = model(batch.to(device))
                # Mask for non-noisy data
                train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                train_labels.append(batch.y.cpu().numpy())  
                train_probs.append(F.softmax(output,dim=1).cpu().numpy())
        
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        if metric=='acc':
            train_acc = accuracy_score(train_labels, train_preds)
        else:
            train_labels_tensor = torch.tensor(train_labels).view(-1,)
            train_preds_tensor = torch.tensor(train_probs[:,1]).view(-1,)
            train_acc = auroc(train_preds_tensor,train_labels_tensor).item()
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for batch in valid_dloader:
                output = model(batch.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        if metric=='acc':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_labels_tensor = torch.tensor(val_labels).view(-1,)
            val_preds_tensor = torch.tensor(val_pred_proba_epoch[:,1]).view(-1,)
            val_acc = auroc(val_preds_tensor,val_labels_tensor).item()
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_acc)
        
        if val_acc > best_valid_acc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = val_acc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc = auroc(test_preds_tensor,test_labels_tensor).item()
                    test_acc_tmp = test_acc
            best_model = deepcopy(model)  # Save the best model
        else:
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc_tmp = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc_tmp = auroc(test_preds_tensor,test_labels_tensor).item()
        print (colored(f'epoch:{epoch},train acc:{train_acc}, val_acc:{val_acc}, test_acc:{test_acc_tmp}','red','on_white'))
        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
    
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'metric_name:{metric}, {model_type}: best_train_acc:{best_train_acc}, best valid_acc:{best_valid_acc}, test_acc:{test_acc}','red','on_yellow'))
    if track_loss:
        return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba,(loss_high_corr,loss_low_corr,loss_dis)
    else:
        return best_model, best_train_acc, best_valid_acc, test_acc, val_pred_proba



def train_model_drugood_no_distillation(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',detectLabelSmoothing=False,block_mask=False,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,sample_entropy_temp=1.0,SL_reg = 1.0,model_type='GIN',reweighting_dict=None,metric='auc',patience=25,q=0.5):
    
    auroc = AUROC(task='binary')
    
    # Initialize containers for top-3 validation and test accuracies
    top_valid_accs = PriorityQueue(maxsize=3)
    top_test_accs = {}
    test_acc_rec = []
    test_acc_rec_top3 = []
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=patience, min_lr=1e-4)

    epochs_since_improvement = 0
    best_valid_acc = -1.0  # Initialize best validation accuracy
    test_acc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for batch in train_dloader:
            optimizer.zero_grad()
            output = model(batch.to(device))
            
            # Handle is_noisy attribute
            svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)

            y = batch.y            
            env_logits = batch.logits
            
            # Handle svm_proba attribute
            y_hat = F.cross_entropy(output, y,reduction='none')
            loss_components = []
            for i in range(env_logits.shape[1]):
                loss_svm = F.cross_entropy(output, env_logits[:,i,:],reduction='none')
                w_ = calculate_sample_weights(env_logits[:,i,:],y,class_num=2,q=q,device=device)
                s_ = torch.mean(y_hat*w_)
                loss_components.append(s_)
            
            loss_components = torch.cat([x.unsqueeze(0) for x in loss_components]).view(-1,)
            #! adaptive KD
            weighted_acc = reweight_valid_acc(svm_valid_accs,temperature=temp).to(device)
            loss_val = torch.sum(weighted_acc*loss_components)
            loss_val.backward()
            optimizer.step()
        
        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for batch in train_dloader:
                output = model(batch.to(device))
                # Mask for non-noisy data
                if "is_noisy" in batch:
                    mask_not_noisy = batch.is_noisy == 0
                    # Select only non-noisy predictions and labels
                    train_preds.append(torch.argmax(output, dim=1)[mask_not_noisy].cpu().numpy())
                    train_labels.append(batch.y[mask_not_noisy].cpu().numpy())
                    train_probs.append(F.softmax(output,dim=1)[mask_not_noisy].cpu().numpy())
                else:
                    train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                    train_labels.append(batch.y.cpu().numpy())  
                    train_probs.append(F.softmax(output,dim=1).cpu().numpy())
        
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        if metric=='acc':
            train_acc = accuracy_score(train_labels, train_preds)
        else:
            train_labels_tensor = torch.tensor(train_labels).view(-1,)
            train_preds_tensor = torch.tensor(train_probs[:,1]).view(-1,)
            train_acc = auroc(train_preds_tensor,train_labels_tensor).item()
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for batch in valid_dloader:
                output = model(batch.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        if metric=='acc':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_labels_tensor = torch.tensor(val_labels).view(-1,)
            val_preds_tensor = torch.tensor(val_pred_proba_epoch[:,1]).view(-1,)
            val_acc = auroc(val_preds_tensor,val_labels_tensor).item()
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_acc)
        
        if val_acc > best_valid_acc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = val_acc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc = auroc(test_preds_tensor,test_labels_tensor).item()
                    test_acc_tmp = test_acc
            best_model = deepcopy(model)  # Save the best model
            
            if top_valid_accs.full():
                # Remove the lowest accuracy if the queue is full
                lowest_val_acc = top_valid_accs.get()
                del top_test_accs[lowest_val_acc]

            top_valid_accs.put(val_acc)
            top_test_accs[best_valid_acc] = test_acc
        else:
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc_tmp = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc_tmp = auroc(test_preds_tensor,test_labels_tensor).item()
        # print (colored(f'epoch:{epoch},train auc:{train_acc}, val_auc:{val_acc}, test_auc:{test_acc_tmp}','red','on_white'))
        test_acc_top3 = max([top_test_accs[acc] for acc in sorted(top_valid_accs.queue, reverse=True)])
        if epoch==99 or epoch==29 or epoch==39 or epoch==49 or epoch==59 or epoch==69 or epoch==79 or epoch==89:
            test_acc_rec.append(test_acc)
            test_acc_rec_top3.append(test_acc_top3)
        
        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'metric_name:{metric}, {model_type}: best_train_acc:{best_train_acc}, best valid_acc:{best_valid_acc}, test_acc:{max(test_acc_rec_top3)}','red','on_yellow'))

    return best_model, best_train_acc, best_valid_acc, test_acc_rec, val_pred_proba,test_acc_rec_top3



def train_model_drugood(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',detectLabelSmoothing=False,block_mask=False,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,sample_entropy_temp=1.0,SL_reg = 1.0,model_type='GIN',reweighting_dict=None,metric='auc',patience=25,q=0.5):
    
    auroc = AUROC(task='binary')
    
    # Initialize containers for top-3 validation and test accuracies
    top_valid_accs = PriorityQueue(maxsize=3)
    top_test_accs = {}
    test_acc_rec = []
    test_acc_rec_top3 = []
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=patience, min_lr=1e-4)

    epochs_since_improvement = 0
    best_valid_acc = -1.0  # Initialize best validation accuracy
    test_acc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for batch in train_dloader:
            optimizer.zero_grad()
            output = model(batch.to(device))
            
            # Handle is_noisy attribute
            svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)

            y = batch.y            
            env_logits = batch.logits
            
            # Handle svm_proba attribute
            y_hat = F.cross_entropy(output, y,reduction='mean')
            if sample_entropy_temp>0:
                env_entropy = calculate_entropy(env_logits)
                env_entropy_weight = softmax_with_temperature_v2(-1.*env_entropy,sample_entropy_temp)  # (N,K) for K svms
            else:
                env_entropy_weight = 1.0
            
            loss_components = []
            for i in range(env_logits.shape[1]):
                loss_svm = F.cross_entropy(output, env_logits[:,i,:],reduction='none')
                w_ = calculate_sample_weights(env_logits[:,i,:],y,class_num=2,q=q,device=device)
                s_ = torch.mean(loss_svm*w_)
                loss_components.append(s_)
            
            loss_components = torch.cat([x.unsqueeze(0) for x in loss_components]).view(-1,)

            weighted_acc = reweight_valid_acc(svm_valid_accs,temperature=temp).to(device)
            loss_val = y_hat + SL_reg*torch.sum(weighted_acc*loss_components)
            loss_val.backward()
            optimizer.step()
        
        
        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for batch in train_dloader:
                output = model(batch.to(device))
                # Mask for non-noisy data
                if "is_noisy" in batch:
                    mask_not_noisy = batch.is_noisy == 0
                    # Select only non-noisy predictions and labels
                    train_preds.append(torch.argmax(output, dim=1)[mask_not_noisy].cpu().numpy())
                    train_labels.append(batch.y[mask_not_noisy].cpu().numpy())
                    train_probs.append(F.softmax(output,dim=1)[mask_not_noisy].cpu().numpy())
                else:
                    train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                    train_labels.append(batch.y.cpu().numpy())  
                    train_probs.append(F.softmax(output,dim=1).cpu().numpy())
        
        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        if metric=='acc':
            train_acc = accuracy_score(train_labels, train_preds)
        else:
            train_labels_tensor = torch.tensor(train_labels).view(-1,)
            train_preds_tensor = torch.tensor(train_probs[:,1]).view(-1,)
            train_acc = auroc(train_preds_tensor,train_labels_tensor).item()
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for batch in valid_dloader:
                output = model(batch.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        if metric=='acc':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_labels_tensor = torch.tensor(val_labels).view(-1,)
            val_preds_tensor = torch.tensor(val_pred_proba_epoch[:,1]).view(-1,)
            val_acc = auroc(val_preds_tensor,val_labels_tensor).item()
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_acc)
        
        if val_acc > best_valid_acc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = val_acc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc = auroc(test_preds_tensor,test_labels_tensor).item()
                    test_acc_tmp = test_acc
            best_model = deepcopy(model)  # Save the best model
            
            if top_valid_accs.full():
                # Remove the lowest accuracy if the queue is full
                lowest_val_acc = top_valid_accs.get()
                del top_test_accs[lowest_val_acc]

            top_valid_accs.put(val_acc)
            top_test_accs[best_valid_acc] = test_acc
        else:
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_probs = []
                with torch.no_grad():
                    for batch in test_dloader:
                        output = model(batch.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_probs.append(F.softmax(output,dim=1).cpu().numpy())
                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_probs = np.concatenate(test_probs, axis=0)
                    if metric=='acc':
                        test_acc_tmp = accuracy_score(test_labels, test_preds)
                    else:
                        test_labels_tensor = torch.tensor(test_labels).view(-1,)
                        test_preds_tensor = torch.tensor(test_probs[:,1]).view(-1,)
                        test_acc_tmp = auroc(test_preds_tensor,test_labels_tensor).item()
        # print (colored(f'epoch:{epoch},train auc:{train_acc}, val_auc:{val_acc}, test_auc:{test_acc_tmp}','red','on_white'))
        test_acc_top3 = max([top_test_accs[acc] for acc in sorted(top_valid_accs.queue, reverse=True)])
        if epoch==99 or epoch==29 or epoch==39 or epoch==49 or epoch==59 or epoch==69 or epoch==79 or epoch==89:
            test_acc_rec.append(test_acc)
            test_acc_rec_top3.append(test_acc_top3)
        
        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'metric_name:{metric}, {model_type}: best_train_acc:{best_train_acc}, best valid_acc:{best_valid_acc}, test_acc:{max(test_acc_rec_top3)}','red','on_yellow'))

    return best_model, best_train_acc, best_valid_acc, test_acc_rec, val_pred_proba,test_acc_rec_top3


def train_model_ogbg(model, train_dloader, valid_dloader, test_dloader, max_epochs, lr,lr_scheduler=None,early_stop_epochs=60,device = 'cpu',evaluator = None,useKD = False,svm_valid_accs = None,num_classes=2,temp=1.0,SL_reg = 1.0,model_type='GIN',reweighting_dict=None,eval_metric='rocauc'):
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler if provided
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=15, min_lr=2e-4)

    epochs_since_improvement = 0
    best_valid_auc = -1.0  # Initialize best validation accuracy
    test_auc = 0.0  # Initialize test accuracy
    val_pred_proba = []  # Initialize validation prediction probabilities
    
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        # Training loop
        if device=='cpu':
            print ('warning, using CPU!')

        model.train()
        for batch in train_dloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            x,edge_index,edge_attr,batch_idx = batch.x,batch.edge_index,batch.edge_attr,batch.batch
            output = model(x.to(device),edge_index.to(device),edge_attr.to(device),batch_idx.to(device))
            if not useKD:
                loss = F.cross_entropy(output,batch.y)
                loss.backward()
                optimizer.step()
            else:
                # Handle is_noisy attribute
                svm_valid_accs = torch.tensor(svm_valid_accs).view(1,-1).to(device)  # shape:(1,5)
                loss_noisy = torch.tensor(0.).to(device)
                loss_not_noisy = torch.tensor(0.).to(device)
                mask_noisy = batch.is_noisy == 1
                mask_not_noisy = batch.is_noisy == 0

                output_noisy = output[mask_noisy]
                output_not_noisy = output[mask_not_noisy]
                
                y_noisy = batch.y[mask_noisy]
                y_not_noisy = batch.y[mask_not_noisy]
                y_not_noisy = y_not_noisy.view(-1,)
                if torch.any(mask_noisy):
                    # Cross-entropy with label smoothing for noisy data
                    loss_noisy = F.cross_entropy(output_noisy, y_noisy,label_smoothing=0.2)
                
                # Regular cross-entropy for not noisy data
                if torch.any(mask_not_noisy):
                    loss_not_noisy = F.cross_entropy(output_not_noisy, y_not_noisy)
                
                # Handle svm_proba attribute
                
                svm_proba = batch.svm_proba  # Assuming shape (N, 5, C)
                svm_proba = svm_proba.view(-1,5,num_classes)
                output_prob = F.log_softmax(output, dim=1).unsqueeze(1)
                loss_svm = torch.sum(-svm_proba * output_prob, dim=2)  # Shape (N, 5)
                
                if reweighting_dict is not None and epoch>=5:
                    # Assuming batch.id is a tensor of shape (N,)
                    reweight_factors_list = [reweighting_dict[i.item()] for i in batch.id]

                    # Convert the list of lists to a 2D tensor of shape (N, 5)
                    reweight_factors = torch.tensor(reweight_factors_list, dtype=torch.float32).to(device)

                    # Apply the reweighting to the loss
                    loss_svm = loss_svm * reweight_factors       
                
                #! adaptive KD
                weighted_acc = reweight_valid_acc(svm_valid_accs,temperature=temp).to(device)
                loss_svm = torch.mean(torch.sum(weighted_acc*loss_svm,dim=1)) # KD loss
                # # Final loss as the average of the 5 losses
                # loss_svm_avg = torch.mean(loss_svm)
                # Combine all losses
                loss = loss_not_noisy + SL_reg*loss_svm
                loss.backward()
                optimizer.step()


        # New block to compute accuracy on train_loader for non-noisy data
        model.eval()  # Set model to evaluation mode
        train_preds = []
        train_labels = []
        with torch.no_grad():
            for batch in train_dloader:
                batch = batch.to(device)
                x,edge_index,edge_attr,batch_idx = batch.x,batch.edge_index,batch.edge_attr,batch.batch
                output = model(x.to(device),edge_index.to(device),edge_attr.to(device),batch_idx.to(device))
                # Mask for non-noisy data
                if "is_noisy" in batch:
                    mask_not_noisy = batch.is_noisy == 0
                    # Select only non-noisy predictions and labels
                    # train_preds.append(torch.argmax(output, dim=1)[mask_not_noisy].cpu().numpy())
                    norm_output = F.softmax(output, dim=1)
                    train_preds.append(norm_output[mask_not_noisy][:,1].cpu().numpy())
                    train_labels.append(batch.y[mask_not_noisy].cpu().view(-1,).numpy())
                else:
                    # train_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                    train_preds.append(norm_output[mask_not_noisy][:,1].cpu().numpy())
                    train_labels.append(batch.y.cpu().view(-1,).numpy())     

        train_preds = np.concatenate(train_preds, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        # train_acc = accuracy_score(train_labels, train_preds)
        train_preds = torch.tensor(train_preds).view(-1,1)
        train_labels = torch.tensor(train_labels).view(-1,1)
        input_dict = {"y_true": train_labels, "y_pred": train_preds}
        train_auc = evaluator.eval(input_dict)[eval_metric]
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        val_pred_proba_epoch = []
        with torch.no_grad():
            for batch in valid_dloader:
                batch = batch.to(device)
                x,edge_index,edge_attr,batch_idx = batch.x,batch.edge_index,batch.edge_attr,batch.batch
                output = model(x.to(device),edge_index.to(device),edge_attr.to(device),batch_idx.to(device))
                val_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
                val_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())


        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_pred_proba_epoch = np.concatenate(val_pred_proba_epoch, axis=0)
        val_pred_proba = torch.tensor(val_pred_proba_epoch)[:,1].view(-1,1)
        val_labels = torch.tensor(val_labels).view(-1,1)
        input_dict = {"y_true": val_labels, "y_pred": val_pred_proba}
        val_auc = evaluator.eval(input_dict)[eval_metric]
        # val_acc = accuracy_score(val_labels, val_preds)
        
        # Update learning rate if scheduler is provided
        if lr_scheduler:
            scheduler.step(val_auc)
        
        if val_auc > best_valid_auc:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        # Check for best validation accuracy
        if val_auc > best_valid_auc:
            best_train_auc = train_auc
            best_valid_auc = val_auc
            val_pred_proba = val_pred_proba_epoch  # Save validation prediction probabilities
            # Test loop
            if test_dloader is not None:
                test_preds = []
                test_labels = []
                test_pred_proba_epoch = []
                with torch.no_grad():
                    for batch in test_dloader:
                        batch = batch.to(device)
                        x,edge_index,edge_attr,batch_idx = batch.x,batch.edge_index,batch.edge_attr,batch.batch
                        output = model(x.to(device),edge_index.to(device),edge_attr.to(device),batch_idx.to(device))
                        test_preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        test_labels.append(batch.y.cpu().numpy())
                        test_pred_proba_epoch.append(F.softmax(output, dim=1).cpu().numpy())


                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_pred_proba_epoch = np.concatenate(test_pred_proba_epoch, axis=0)
                    test_pred_proba = torch.tensor(test_pred_proba_epoch)[:,1].view(-1,1)
                    test_labels = torch.tensor(test_labels).view(-1,1)
                    input_dict = {"y_true": test_labels, "y_pred": test_preds}
                    # test_acc = accuracy_score(test_labels, test_preds)
                    test_preds = torch.tensor(test_pred_proba)
                    test_labels = torch.tensor(test_labels)
                    input_dict = {"y_true": test_labels, "y_pred": test_preds}
                    test_auc = evaluator.eval(input_dict)[eval_metric]
                    
                    
            best_model = deepcopy(model)  # Save the best model

        if epochs_since_improvement >= early_stop_epochs:
            print("Early stopping triggered.")
            time.sleep(1)
            break
    
        # print(f"Epoch {epoch+1}/{max_epochs} - Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")
    model.train()
    print (colored(f'{model_type}: best_train_acc(only consider labelled data):{best_train_auc}, best valid_acc:{best_valid_auc}, test_acc:{test_auc}','red','on_yellow'))
    return best_model, best_train_auc, best_valid_auc, test_auc, val_pred_proba

    
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from copy import deepcopy

def finetune_train(model, train_loader, val_loader, test_loader, max_epochs, early_stop_epoch, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    test_acc = 0
    best_model = None
    early_stop_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        train_acc = finetune_evaluate(model, train_loader)
        val_acc = finetune_evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = finetune_evaluate(model, test_loader)
            best_model = deepcopy(model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1    
        if early_stop_counter >= early_stop_epoch:
            break
        print (colored(f'epoch:{epoch}, train_acc:{train_acc}, val_acc:{val_acc},best_val_acc:{best_val_acc}, test_acc:{test_acc}','blue','on_white'))
    return best_model,train_acc, best_val_acc, test_acc

def finetune_evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# Example usage
# model = ...  # Initialize your model
# train_loader, val_loader, test_loader = ...  # Initialize your data loaders
# max_epochs, early_stop_epoch, lr = ...  # Set your training parameters
# lr_scheduler = ...  # Define
   