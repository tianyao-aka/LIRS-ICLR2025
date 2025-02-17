
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
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from utils.functionalUtils import *
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import entropy
from torch_geometric.data import Data,DataLoader


class ALPolicy:
    """
    Active Learning Policy class to manage labelled and unlabelled indices.
    
    Attributes:
        train_labelled_indices (torch.Tensor): Tensor of indices that are labelled.
        train_unlabelled_indices (torch.Tensor): Tensor of indices that are unlabelled.
    """
    
    def __init__(self, train_labelled_indices, train_unlabelled_indices,K,dataset,graph_emb,val_indices,test_indices=None,epochs = 50,topK_svm=50,topK_gamlp=50,device = 0):
        """
        Initialize the ALPolicy object.
        
        Parameters:
            train_labelled_indices (torch.Tensor): Initial tensor of labelled indices.
            train_unlabelled_indices (torch.Tensor): Initial tensor of unlabelled indices.
            K: number samples per iteration
        """
        
        self.train_labelled_indices = train_labelled_indices
        self.train_unlabelled_indices = train_unlabelled_indices
        self.K = K
        self.dataset = dataset
        self.graph_emb = graph_emb
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.epochs = epochs
        self.topK_svm = topK_svm
        self.topK_gamlp = topK_gamlp
        self.device = device
        
        
    def union_tensors(self):
        """
        Union two 1D tensors and return a 1D tensor.
        
        Parameters:
        - tensor1 (torch.Tensor): A 1D tensor.
        - tensor2 (torch.Tensor): A 1D tensor.
        
        Returns:
        - torch.Tensor: A 1D tensor containing the union of tensor1 and tensor2.
        """
        
        # Concatenate the two tensors
        t1 = self.train_labelled_indices
        t2 = self.train_unlabelled_indices
        concatenated_tensor = torch.cat((t1,t2))
        return concatenated_tensor
    
    def update(self, indices_next_iter):
        """
        Update the labelled and unlabelled indices for the next iteration.
        
        Parameters:
            indices_next_iter (torch.Tensor): Tensor of indices to be labelled in the next iteration.
        """
        # Remove the indices from the unlabelled set
        mask = torch.isin(self.train_unlabelled_indices, indices_next_iter, invert=True)
        self.train_unlabelled_indices = self.train_unlabelled_indices[mask]
        
        # Add the indices to the labelled set
        self.train_labelled_indices = torch.cat((self.train_labelled_indices, indices_next_iter))


    def random_policy(self):
        """
        Implement a random policy for selecting the next K indices to label.
        
        Parameters:
            K (int): The number of indices to randomly select for labelling.
            
        Returns:
            torch.Tensor: A tensor containing the randomly selected indices.
        """
        
        # Randomly select K indices from the unlabelled set
        random_indices = torch.randperm(len(self.train_unlabelled_indices))[:self.K]
        selected_indices = self.train_unlabelled_indices[random_indices]
        # Update the labelled and unlabelled indices
        self.update(selected_indices)

    

    def _get_most_uncertain_samples_from_svm(self, svm_emb_list):
        """
        Get the top-K most uncertain samples and their indices based on the average predicted probabilities from multiple SVMs.
        
        Parameters:
        svm_emb_list (list): List of tuples containing SVM models and their corresponding embeddings.
        
        Returns:
        list: Top-K most uncertain samples and their indices, sorted by their entropy in descending order.
        """
        
        unlabelled_indices = self.train_unlabelled_indices
        prob_list = []
        
        # Collect probabilities from each SVM model
        for svm, emb in svm_emb_list:
            probs = svm.predict_proba(emb[unlabelled_indices])
            reshaped_probs = np.expand_dims(probs, axis=0)  # Reshape to (1, N, C)
            prob_list.append(reshaped_probs)

        # Concatenate along a new axis to get an array of shape (L, N, C)
        prob_array = np.concatenate(prob_list, axis=0)
        
        # Calculate the mean over L to get an array of shape (N, C)
        avg_probs = np.mean(prob_array, axis=0)
        
        # Calculate the entropy for each sample
        entropies = [entropy(p, base=2) for p in avg_probs]
        
        # Sort the samples by their entropy
        sorted_indices = np.argsort(entropies)[::-1]
        
        # Get the top-K most uncertain samples
        topK_uncertain_indices = [unlabelled_indices[i].item() for i in sorted_indices[:self.topK_svm]]
        
        return topK_uncertain_indices,svm_emb_list[0][1][unlabelled_indices],entropies  # emb corresponding to the best svm model



    def _get_most_uncertain_samples_from_gamlp(self,model,num_infer=10):
        """
        Get the top-K most uncertain samples and their indices.
        
        Parameters:
        model (torch.nn.Module): The PyTorch model for inference.
        unlabelled_indices (list): List of indices corresponding to the unlabelled samples.
        dataset (torch_geometric.data.Dataset): The PyG dataset.
        device (str): The device to use for computation ('cuda' or 'cpu').
        K (int): The number of top uncertain samples to return.
        
        Returns:
        list: Top-K most uncertain samples' indices, sorted by their entropy in descending order.
        """
        
        device = self.device
        K = self.topK_gamlp
        
        # Step 1: Create DataLoader from sub-dataset
        sub_dataset = self.dataset[self.train_unlabelled_indices]
        data_loader = DataLoader(sub_dataset, shuffle=False,batch_size=128,num_workers=1)
        
        # Initialize a list to store the probabilities for each inference run
        prob_list = []
        avg_g_embs = []
        
        # Step 2: Forward computation and probability normalization
        model.eval()
        with torch.no_grad():
            for i in range(num_infer):  # Perform inference 10 times
                batch_probs = []
                g_embs = []
                for batch in data_loader:
                    batch = batch.to(device)
                    if num_infer>1:
                        logits,_,g_emb = model(batch, mask=True,output_emb=True)
                    else:
                        logits,_,g_emb = model(batch, output_emb=True)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    batch_probs.append(probs)
                    g_embs.append(g_emb)
                
                # Concatenate batch probabilities and reshape to (1, N, C)
                batch_probs = np.concatenate(batch_probs, axis=0)
                reshaped_probs = np.expand_dims(batch_probs, axis=0)
                prob_list.append(reshaped_probs)
                
                g_embs = torch.cat(g_embs,dim=0).unsqueeze(0)
                avg_g_embs.append(g_embs)
            
            # Concatenate along a new axis to get an array of shape (10, N, C)
            prob_array = np.concatenate(prob_list, axis=0)
            
            # Calculate the mean over the first axis to get an array of shape (N, C)
            avg_probs = np.mean(prob_array, axis=0)
            # calc graph avg embs
            avg_g_embs = torch.mean(torch.cat(avg_g_embs,dim=0),dim=0)
            
        # Step 3: Calculate entropy and get top-K uncertain samples
        entropies = [entropy(p, base=2) for p in avg_probs]
        sorted_indices = np.argsort(entropies)[::-1]
        topK_uncertain_indices = [self.train_unlabelled_indices[i].item() for i in sorted_indices[:K]]
        model.train()
        return topK_uncertain_indices,avg_g_embs,entropies

    def get_most_uncertain_samples(self,svms,ga_mlp):
        from_svm,svm_emb,svm_entropy = self._get_most_uncertain_samples_from_svm(svms)
        from_gamlp,gamlp_emb,gamlp_entropy = self._get_most_uncertain_samples_from_gamlp(ga_mlp)
        print (colored(f"from svm:{from_svm}", 'blue','on_white'))
        print (colored(f"from gamlp:{from_gamlp}", 'blue','on_white'))
        return from_svm,from_gamlp
    
    def dpp_sampling(self,svms,ga_mlp,block_mask = False):
        """
        Get the top-K samples using k-DPP
        
        Parameters:
        svms (List of Lists): each list contains, (clf,graph_emb)
        ga_mlp (nn.Module): a torch model of ga-mlp
        """
        num_infer = 1 if not block_mask else 10
        num_fail_try=5
        retires = 0
        best_det_val = -1.
        while retires<num_fail_try:
            try:
                svm_index,svm_emb,svm_entropy = self._get_most_uncertain_samples_from_svm(svms)
                # gamlp_index,gamlp_emb,gamlp_entropy = self._get_most_uncertain_samples_from_gamlp(ga_mlp,num_infer=num_infer)
                
                L_svm,union_indices = self._calc_L(svm_index, svm_index, svm_emb, svm_entropy)
                # L_gamlp,_ = self._calc_L(svm_index, gamlp_index, gamlp_emb, gamlp_entropy)
                L = L_svm
                
                # rank = np.linalg.matrix_rank(L_svm)
                # print ('rank:',rank)
                DPP = FiniteDPP('likelihood', **{'L': L})
                DPP.sample_mcmc_k_dpp(size=self.K)
                break
            except Exception as e:
                print (f'fail,retry at {retires}.')
                retires +=1
                continue
        if retires==num_fail_try:
            raise ValueError("k-dpp sampling error")
        
        sample_list = DPP.list_of_samples # the index here is the index in L, not consistent with union_indices
        sample_list = remove_duplicates(sample_list[0])
        for idx,s in enumerate(sample_list):
            det_val = submatrix_det(L,s)
            if det_val>best_det_val:
                best_det_val = det_val
                best_idx = idx
        best_sample = sample_list[best_idx]
        best_sample = torch.tensor([union_indices[i] for i in best_sample]) # get the final indices from unlabelled pool
        # print ('best det:',best_det_val)
        # print ('best sample:',best_sample)
        # print (colored(f"dpp sample list:{sample_list}", 'blue','on_white'))
        self.update(best_sample)
    
    
    def _calc_L(self,from_svm, from_ga_mlp, emb, entropy):
        
        """
        Calculate matrices Q and S for SVM.
        
        Parameters:
        - from_svm: List of indices corresponding to SVM.
        - from_ga_mlp: List of indices corresponding to GA-MLP.
        - emb: PyTorch tensor containing embeddings for models. Shape: (n, d), where n is the number of samples and d is the dimensionality.
        - entropy: NumPy array containing entropy values for model. Shape: (n,), where n is the number of samples.
        - unlabelled_train_indices: 1-D PyTorch tensor containing indices for unlabelled training data.
        
        Returns:
        - Q: NumPy array, shape (N, N), where N is the number of unique indices in the union of from_svm and from_ga_mlp. Diagonal elements are populated based on svm_entropy.
        - S: NumPy array, shape (N, N), where N is the number of unique indices in the union of from_svm and from_ga_mlp. Elements are populated based on the pairwise dot products of embeddings in svm_emb.
        """
        
        # Step 1: Get the union of from_svm and from_ga_mlp and sort it
        unlabelled_train_indices = self.train_unlabelled_indices
        union_indices = list(set(from_svm + from_ga_mlp))
        union_indices.sort()
        N = len(union_indices)
        # Create a mapping from unlabelled_train_indices to row indices in svm_emb and svm_entropy
        index_map = {idx: i for i, idx in enumerate(unlabelled_train_indices.tolist())}
        
        # Extract corresponding embeddings
        new_emb = torch.stack([emb[index_map[idx]] for idx in union_indices])
        # Initialize new_svm_entropy with zeros
        noise = torch.normal(mean=0., std=0.001, size=new_emb.shape)
        new_emb = new_emb + noise
        new_entropy = np.zeros(N)
        
        
        # Populate new_svm_entropy for indices in from_svm
        for idx in union_indices:
            row = union_indices.index(idx)
            new_entropy[row] = entropy[index_map[idx]]+1e-6
        
        # Step 2: Initialize Q with diagonal as new_svm_entropy
        Q = np.zeros((N, N))*1.0
        np.fill_diagonal(Q, new_entropy)
        
        # Step 3: Initialize S and calculate pairwise dot products
        S = torch.mm(new_emb, new_emb.t()).cpu().numpy()
        # if not is_psd(S):
        #     print (colored(f"2 S is not PSD", 'yellow','on_blue'))
        QSQ = np.dot(np.dot(Q, S), Q)
        #! debug psd issues
        # if check_tensor_for_inf_nan(torch.tensor(Q)):
        #     print (colored(f"Q contains inf or nan", 'yellow','on_blue'))
        # if check_tensor_for_inf_nan(torch.tensor(S)):
        #     print (colored(f"S contains inf or nan", 'yellow','on_blue'))
        # if not is_psd(S):
        #     print (colored(f"S is not PSD", 'yellow','on_blue'))
        # if not is_symmetric(S):
        #     print (colored(f"S is not symmetric", 'yellow','on_blue'))
        
        
        return QSQ,union_indices

    
    
    def train_and_evaluate_svm_with_proba(self,C=10.0,cv=3):
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
        
        dataset = self.dataset
        graph_emb = self.graph_emb
        val_indices = self.val_indices
        test_indices = self.test_indices
        labeled_train_indices = self.train_labelled_indices
        
        # Get the embeddings and labels for the training set
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
            
            return val_accuracy, val_proba, test_accuracy
        
        return val_accuracy, val_proba, None



def check_tensor_for_inf_nan(tensor):
    """
    Checks if a given PyTorch tensor contains any 'inf' or 'nan' values.
    
    Parameters:
    - tensor (torch.Tensor): The input tensor to be checked.
    
    Returns:
    - bool: True if the tensor contains 'inf' or 'nan', False otherwise.
    """
    
    # Check for 'inf' values
    contains_inf = torch.isinf(tensor).any()
    
    # Check for 'nan' values
    contains_nan = torch.isnan(tensor).any()
    
    return contains_inf or contains_nan


def is_psd(matrix):
    """
    Checks if a given NumPy array is a Positive Semi-Definite (PSD) matrix.
    
    Parameters:
    - matrix (np.ndarray): The input square matrix to be checked.
    
    Returns:
    - bool: True if the matrix is PSD, False otherwise.
    """
    
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if the matrix is Hermitian (conjugate transpose equals itself)
    if not np.allclose(matrix, matrix.conj().T):
        return False
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    # Check if all eigenvalues are non-negative
    if np.any(eigenvalues < 0):
        print (eigenvalues)
        return False
    
    return True

def is_symmetric(matrix):
    """
    Checks if a given NumPy array is a symmetric matrix.
    
    Parameters:
    - matrix (np.ndarray): The input square matrix to be checked.
    
    Returns:
    - bool: True if the matrix is symmetric, False otherwise.
    """
    
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if the matrix is equal to its transpose
    return np.allclose(matrix, matrix.T)



def check_and_adjust_PSD(S, tol=1e-12):
    """
    Checks if a matrix is PSD and adjusts eigenvalues if they are close to zero.
    
    Parameters:
    - S (np.ndarray): The square matrix to be checked.
    - tol (float): Tolerance level for eigenvalues close to zero.
    
    Returns:
    - np.ndarray: Adjusted PSD matrix.
    """
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    
    # Check for eigenvalues close to zero and set them to zero
    eigenvalues[eigenvalues < tol] = 0.
    # Reconstruct the matrix
    S_adjusted = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return S_adjusted

# Example usage
if __name__ == "__main__":
    al_policy = ALPolicy(train_labelled_indices=[1, 2, 3], train_unlabelled_indices=[4, 5, 6, 7, 8])
    
    # Update the policy with new labelled indices
    al_policy.update([4, 5])
    
    # Use random policy to get an index for labelling
    next_index = al_policy.random_policy()
    print(f"Next index to label: {next_index}")


