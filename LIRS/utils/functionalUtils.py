
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
import os
import datetime
import random
from collections import defaultdict
import copy
from torch_geometric.data import DataLoader
from scipy.stats import entropy


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


def scoring(x, q):
    """
    Compute the function (1 - x^q) / q.

    Parameters:
    x (torch.Tensor): A PyTorch tensor.
    q (float): A scalar value.

    Returns:
    torch.Tensor: The result of the function applied element-wise to x.
    """
    if q == 0:
        raise ValueError("q should not be zero to avoid division by zero.")

    return (1 - x.pow(q)) / q

def calculate_sample_weights(logits, Y, class_num, q,device):
    """
    Calculate weights for each sample based on logits and labels.

    Parameters:
    logits (torch.Tensor): Tensor of logits of shape (N, C).
    Y (torch.Tensor): Tensor of labels of shape (N,).
    class_num (int): Number of classes.
    score (function): A function that takes a tensor and returns a tensor of weights.

    Returns:
    torch.Tensor: A tensor of weights of shape (N,).
    """

    N = logits.shape[0]
    weights = torch.zeros(N).to(device)

    for c in range(class_num):
        # Create a mask for the current class
        mask = (Y == c)
        # Select the logits for the current class
        logits_for_class = logits[mask, c]
        # Apply the score function to get weights for the current class
        weights_for_class = logits_for_class
        # Assign the weights to the corresponding positions in the overall weights tensor
        weights[mask] = weights_for_class
    return scoring(weights,q)

# Example usage:
# logits = torch.randn(N, C)  # Replace N and C with actual values
# Y = torch.randint(0, C, (N,))  # Example labels
# class_num = C  # Number of classes
# w_ = calculate_weights(logits, Y, class_num, score_function)  # score_function should be defined


def get_svm_train_logits(clf, train_emb):

    # Step 1: Convert the PyTorch tensor to a NumPy array
    train_emb_np = train_emb.numpy()

    # Step 2: Use the classifier to predict the probability matrix
    proba_matrix = clf.predict_proba(train_emb_np)

    # Step 3: Convert the probability matrix back to a PyTorch tensor and return it
    proba_tensor = torch.tensor(proba_matrix).float()

    return proba_tensor


def save_pdf_files(pdf_list, folder_name):
    """
    Creates a directory and saves PDF files in it.

    Parameters:
    - pdf_list: List[Tuple[float, bytes]]
        A list where each item is a tuple containing a training accuracy and a PDF file in bytes.
    - folder_name: str
        The name of the folder to be created for saving PDFs.
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save each PDF file in the directory
    for train_acc, pdf_bytes in pdf_list:
        file_path = os.path.join(folder_name, f"train_acc_{train_acc:.5f}.pdf")
        with open(file_path, 'wb') as file:
            file.write(pdf_bytes)


def extract_embeddings(M, indices_list, dataset,device):
    """
    Extract embeddings for specified indices using a given model and dataset.
    
    Parameters:
    - M (torch.nn.Module): The PyTorch model.
    - indices_list (list of torch.Tensor): A list containing 4 1D tensors, each indicating indices.
    - dataset (torch_geometric.data.Dataset): The PyG dataset.
    
    Returns:
    - List of numpy arrays: A list containing 4 numpy arrays, each representing embeddings for the specified indices.
    """
    M.to(device)
    # Create a data loader for the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Initialize an empty list to store all embeddings
    all_embeddings = []
    
    # Set the model to evaluation mode and disable gradient computation
    M.eval()
    with torch.no_grad():
        for batch in data_loader:
            # Get embeddings for the current batch
            _,_,batch_embeddings = M(batch.to(device), output_emb=True)
            all_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    # Extract embeddings for the specified indices and cast to numpy arrays
    numpy_arrays = [all_embeddings_tensor[indices].cpu().numpy() for indices in indices_list]
    
    return numpy_arrays



def interpolate_models(M1, M2, alpha=0.9, noise_std=0.01,device = 0):
    """
    Create a new model by interpolating the parameters of two given models M1 and M2.
    
    Parameters:
    - M1 (nn.Module): The first PyTorch model.
    - M2 (nn.Module): The second PyTorch model.
    - alpha (float): The weight for the first model's parameters.
    - noise_std (float): The standard deviation of the Gaussian noise to be added.
    
    Returns:
    nn.Module: A new PyTorch model with interpolated parameters.
    """
    
    # Deep copy one of the models to serve as the starting point for the new model
    new_M = copy.deepcopy(M1)
    
    # Get the state dictionaries of the models
    state_dict_M1 = M1.state_dict()
    state_dict_M2 = M2.state_dict()
    state_dict_new_M = new_M.state_dict()
    
    # Interpolate the parameters
    for key in state_dict_M1:
        # Interpolate the parameters and add noise
        state_dict_new_M[key] = alpha * state_dict_M1[key] + (1 - alpha) * state_dict_M2[key]
        noise = torch.normal(mean=0., std=noise_std, size=state_dict_new_M[key].size()).to(device)
        state_dict_new_M[key] += noise
    
    # Load the interpolated parameters into the new model
    new_M.load_state_dict(state_dict_new_M)
    
    return new_M

def check_elements_in_list(l, K):
    """
    Checks if all elements in list 'l' are present in list 'K'.
    
    Parameters:
    - l (List[Any]): A list containing elements to be checked.
    - K (List[Any]): A reference list.
    
    Returns:
    bool: True if all elements in 'l' are in 'K', otherwise False.
    """
    
    # Convert both lists to sets for efficient membership checking
    set_l = set(l)
    set_K = set(K)
    
    # Check if all elements in set_l are in set_K
    return set_l.issubset(set_K)

def submatrix_det(L,sublist):
    """
    Given a list of indices, a sublist which is a subset of indices, and a 2D NumPy array L,
    this function returns the determinant of the submatrix of L corresponding to the rows
    in sublist.
    
    Parameters:
    - sublist (List[int]): A list containing row indices.
    - L (np.ndarray): A 2D NumPy array where each row corresponds to an index in 'indices'.
    
    Returns:
    float: The determinant of the submatrix of L corresponding to the rows in 'sublist'.
    """
    
    # Extract the submatrix from L using the row indices
    sub_L = L[np.ix_(sublist, sublist)]
    
    # Compute the determinant of the submatrix
    det_value = np.linalg.det(sub_L)
    
    return det_value

def remove_duplicates(lst_of_lsts):
    """
    Given a list of lists, each containing K integers, this function sorts each inner list,
    removes duplicate inner lists, and returns the list of unique lists.
    
    Parameters:
    lst_of_lsts (List[List[int]]): A list of lists, each containing K integers.
    
    Returns:
    List[List[int]]: A list of unique lists, each sorted in ascending order.
    """
    
    # Sort each inner list
    sorted_lsts = [sorted(inner_lst) for inner_lst in lst_of_lsts]
    
    # Convert each inner list to a tuple so it can be hashed, then remove duplicates
    unique_lsts = list(set(tuple(inner_lst) for inner_lst in sorted_lsts))
    
    # Convert each inner tuple back to a list
    unique_lsts = [list(inner_lst) for inner_lst in unique_lsts]
    
    return unique_lsts

def find_top_key(input_dict):
    """
    Find the top-10 'k' with the best 'acc' across all 'c'.
    
    Parameters:
    input_dict (dict): Dictionary with keys as (k, c) and values as acc.
    
    Returns:
    list: Top-10 'k' sorted by the best 'acc' in descending order.
    """
    
    # Step 1: Find the best 'acc' for each 'k' across all 'c'
    best_acc_for_k = defaultdict(float)  # Default value is 0.0 for each new 'k'
    for (k, c), acc in input_dict.items():
        if acc > best_acc_for_k[k]:
            best_acc_for_k[k] = acc
    
    # Step 2: Sort 'k' based on best 'acc' in descending order
    sorted_k = sorted(best_acc_for_k.keys(), key=lambda x: best_acc_for_k[x], reverse=True)
    
    # Step 3: Get the top-10 'k'
    top_10_k = sorted_k[:10]
    
    return top_10_k


def save_tensor_to_file(X, fpath, name):
    """
    Save a PyTorch tensor X to a specified file path and name.
    
    Parameters:
    - X (torch.Tensor): Tensor of shape (N, D) to be saved.
    - fpath (str): The directory where the tensor should be saved.
    - name (str): The name of the file to save the tensor as.
    
    Returns:
    None
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    # Full path to the file
    full_path = os.path.join(fpath, name)
    
    # Remove the file if it already exists
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Save the tensor to the file
    torch.save(X, full_path)


def pack_arrays_to_3D(array_list):
    """
    Convert a list of 2D NumPy arrays into a 3D NumPy array.
    
    Parameters:
    - array_list (list of np.ndarray): The input list where each element is a 2D NumPy array with shape (N, D).
    
    Output:
    - np.ndarray: A 3D NumPy array with shape (L, N, D).
    """
    
    # Stack the 2D arrays along a new axis to create a 3D array
    array_3D = np.stack(array_list, axis=0)
    
    return array_3D



def pack_arrays_to_4D(nested_list):
    """
    Convert a nested list of 2D NumPy arrays into a 4D NumPy array.
    
    Parameters:
    - nested_list (list of list of np.ndarray): The input nested list where each element is a 2D NumPy array with shape (N, D).
    
    Output:
    - np.ndarray: A 4D NumPy array with shape (K, L, N, D).
    """
    
    # Initialize an empty list to hold the 3D arrays
    list_of_3D_arrays = []
    
    # Loop through the outer list to create 3D arrays
    for inner_list in nested_list:
        # Stack the 2D arrays along a new axis to create a 3D array
        array_3D = np.stack(inner_list, axis=0)
        list_of_3D_arrays.append(array_3D)
    
    # Stack the 3D arrays along a new axis to create a 4D array
    array_4D = np.stack(list_of_3D_arrays, axis=0)
    
    return array_4D


def generate_string():
    """
    Generate a string in the format 'xxxxyyyyyyyy'.
    
    xxxx: Four random digits from 0-9.
    yyyyyyyy: Current date and time in the format 'MMddHHmm'.
    
    Returns:
        str: The generated string.
    """
    
    # Generate four random digits
    random_digits = ''.join([str(random.randint(0, 9)) for _ in range(4)])
    
    # Get the current date and time
    current_datetime = datetime.now().strftime('%m%d%H%M')
    
    # Concatenate the random digits and current date-time
    result = random_digits + current_datetime
    
    return result



def save_numpy_array_to_file(array, file_path, file_name):
    """
    Save a NumPy array to a specified file path and file name.
    
    Parameters:
    - array (np.ndarray): The NumPy array to be saved.
    - file_path (str): The directory where the file will be saved.
    - file_name (str): The name of the file (without extension).
    
    Output:
    - None: The function saves the array to a .npy file at the specified location.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Full path to the file
    full_file_path = os.path.join(file_path, f"{file_name}.npy")
    
    # Delete the file if it already exists
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
    
    # Save the NumPy array to the file
    np.save(full_file_path, array)
    


def write_results_to_file(fpath, n, s):
    # Check if the directory exists, if not, create it
    # fpath: 存放路径
    # n: 文件名 （.txt结尾）
    # s: 内容
    if not os.path.exists(fpath):
        try:
            os.makedirs(fpath)
        except:
            pass

    # Construct full file path
    full_path = os.path.join(fpath, n)

    # Open the file in write mode, which will create the file if it does not exist
    # and overwrite it if it does. Then write the string to the file.
    with open(full_path, 'w') as f:
        f.write(s)


def get_free_gpu(thres): # in GB
    # Initialize NVML
    gpu_list = []
    try:
        nvmlInit()
    except Exception as e:
        return -1
    # Get number of available GPU devices
    device_count = nvmlDeviceGetCount()

    for i in range(device_count):
        # Get handle for the GPU device
        handle = nvmlDeviceGetHandleByIndex(i)
        
        # Get memory info for the GPU device
        mem_info = nvmlDeviceGetMemoryInfo(handle)

        # Check if the free memory is greater than 20 GB
        if mem_info.free / (1024 ** 3) > thres:  # convert bytes to gigabytes
            # Return the device ID
            gpu_list.append((mem_info.free / (1024 ** 3),i))
    if len(gpu_list)==0: 
        # If no GPU with sufficient memory was found, return None
        return None
    else:
        gpu_list = sorted(gpu_list,key=lambda x:x[0],reverse=True)
        print (colored(f'using device {gpu_list[0][1]}, free GPU memory is: {gpu_list[0][0]}','red','on_white'))
        return gpu_list[0][1]
    
    
def calculate_binary_correlation(list1, list2):
    # Convert lists to NumPy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    # Calculate correlation using SciPy
    correlation_scipy, _ = pearsonr(arr1, arr2)
    return correlation_scipy



def compute_average_score(X, Y, K):
    # Normalize each row in X
    # X_norm = F.normalize(X, p=2, dim=1)
    X_norm = X
    # Compute pairwise cosine similarity
    S = torch.mm(X_norm, X_norm.t())
    
    # Initialize total score
    total_score = 0.0
    
    # Loop through each sample in X
    for i in range(X.size(0)):
        # Get the top-K most similar samples for X[i]
        _, topk_indices = torch.topk(S[i], K + 1)  # +1 to include the sample itself
        
        # Remove the sample itself from top-K
        topk_indices = topk_indices[topk_indices != i]
        
        # Count the number of samples with the same label as X[i]
        count_same_label = torch.sum(Y[topk_indices] == Y[i]).float()
        print (f'for sample {i} with label {Y[i]}, the nearest index is {topk_indices}, label s are {Y[topk_indices]}')
        # Calculate the score for this sample
        score = count_same_label / K
        
        # Accumulate the score
        total_score += score
    
    # Calculate the average score
    average_score = total_score / X.size(0)
    
    return average_score.item()



def visualize_with_umap(X, Y, output_file='umap_visualization.pdf'):
    """
    Apply UMAP on the feature matrix X to reduce its dimension to 2D.
    Then visualize the data points in a scatter plot, colored by their labels in Y.
    Save the plot as a PDF file.

    Parameters:
    - X: numpy.ndarray
        The feature matrix of shape (n_samples, n_features).
    - Y: numpy.ndarray
        The label vector of shape (n_samples,).
    - output_file: str
        The name of the output PDF file.

    Returns:
    None
    """
    # Apply UMAP to reduce dimension to 2D
    reducer = umap.UMAP(n_components=2)
    X_reduced = reducer.fit_transform(X)

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Plot points with label 0 as red and label 1 as blue
    plt.scatter(X_reduced[Y == 0, 0], X_reduced[Y == 0, 1], c='red', label='Label 0')
    plt.scatter(X_reduced[Y == 1, 0], X_reduced[Y == 1, 1], c='blue', label='Label 1')

    plt.xlabel('UMAP 1st Component')
    plt.ylabel('UMAP 2nd Component')
    plt.legend()
    plt.title('2D UMAP Visualization')

    # Save the plot as a PDF
    plt.savefig(output_file)


def run_kmeans_with_initial_points(X, Y,indices_0=None,indices_1=None):
    """
    Run KMeans clustering on the feature matrix X using initial points based on class averages.

    Parameters:
    - X: numpy.ndarray
        The feature matrix of shape (N, D).
    - Y: numpy.ndarray
        The label vector of shape (N,).

    Returns:
    - centroids: numpy.ndarray
        The centroids of the two clusters, shape (2, D).
    """
    # Step 1: Sample 5 data points from each class
    if indices_0 is None and indices_1 is None:
        indices_0 = np.random.choice(np.where(Y == 0)[0], size=5, replace=False)
        indices_1 = np.random.choice(np.where(Y == 1)[0], size=5, replace=False)
    else:
        pass

    # Step 2: Compute the average vectors for these points
    initial_centroid_0 = np.mean(X[indices_0], axis=0)
    initial_centroid_1 = np.mean(X[indices_1], axis=0)
    initial_centroids = np.vstack([initial_centroid_0, initial_centroid_1])

    # Step 3: Run KMeans
    kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1)
    kmeans.fit(X)

    # Step 4: Return the centroids
    centroids = kmeans.cluster_centers_
    return centroids


def find_closest_points(X, Y, centroids, K):
    """
    Find the top-K closest points in X to each centroid and return their labels.

    Parameters:
    - X: numpy.ndarray
        The feature matrix of shape (N, D).
    - Y: numpy.ndarray
        The label vector of shape (N,).
    - centroids: numpy.ndarray
        The centroids of the two clusters, shape (2, D).
    - K: int
        The number of closest points to find for each centroid.

    Returns:
    - closest_labels: list of tuples
        The labels of the 2K closest points, in the format [(0,0,1), (1,1,0)].
    - distance_ratios: list of tuples
        The distance ratios of the 2K closest points, in the format [(0.12,0.32,0.11), (0.32,0.45,0.41)].
    """
    closest_labels = []
    distance_ratios = []

    for i, centroid in enumerate(centroids):
        # Compute Euclidean distance to each centroid
        distances = np.linalg.norm(X - centroid, axis=1)
        
        # Find the indices of the top-K closest points
        closest_indices = np.argsort(distances)[:K]
        
        # Get the labels of these points
        closest_labels.append(tuple(Y[closest_indices]))

        # Compute distance ratios for these points
        other_centroid = centroids[1 - i]
        other_distances = np.linalg.norm(X[closest_indices] - other_centroid, axis=1)
        ratios = distances[closest_indices] / other_distances
        distance_ratios.append(tuple(ratios))

    return closest_labels, distance_ratios


def calculate_accuracy(predictions_list):
    """
    Calculate the accuracy of a list of prediction tuples.
    
    Parameters:
    - predictions_list (list of tuples): Each tuple contains a label as its first element 
                                         followed by predicted values. 
                                         For example, (0, 0, 0, 1, 0) where 0 is the label.
                                         
    Returns:
    - float: The accuracy of the predictions.
    
    Formula for Accuracy:
    Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
    """
    
    # Initialize variables to keep track of correct and total predictions
    correct_predictions = 0
    total_predictions = 0
    
    # Iterate through each tuple in the list
    for pred_tuple in predictions_list:
        # Extract the label and the predictions from the tuple
        label = pred_tuple[0]
        predictions = pred_tuple[1:]
        
        # Update the count of correct and total predictions
        correct_predictions += sum([1 for p in predictions if p == label])
        total_predictions += len(predictions)
        
    # Calculate accuracy
    if total_predictions == 0:
        return 0.0  # To avoid division by zero
    else:
        accuracy = correct_predictions / total_predictions
    
    return accuracy



def last_K_unique_elements(lst,K=10):
    """
    Given a list `lst`, this function returns a new list containing the last three non-duplicate elements.
    
    Parameters:
    lst (list): The input list which may contain duplicate elements.
    
    Returns:
    list: A new list containing the last three non-duplicate elements. If there are fewer than three unique elements, the new list will contain all the unique elements.
    """
    
    unique_elements = set()  # To keep track of unique elements encountered
    result = []  # To store the last three unique elements
    
    # Traverse the list in reverse order
    for elem in reversed(lst):
        if elem not in unique_elements:
            unique_elements.add(elem)
            result.append(elem)
        
        # Stop if we've collected the last three unique elements
        if len(result) == K:
            break
            
    return list(reversed(result))


def tensor_difference(a, b):
    """
    Given two 1D PyTorch tensors `a` and `b` where `a` is a subset of `b`,
    this function returns a 1D tensor containing elements in `b` but not in `a`.
    
    Parameters:
    - a (torch.Tensor): A 1D tensor, which is a subset of `b`.
    - b (torch.Tensor): A 1D tensor containing all elements in `a` and possibly more.
    
    Returns:
    - torch.Tensor: A 1D tensor containing elements that are in `b` but not in `a`.
    """
    
    # Convert tensors to NumPy arrays for set operations
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    
    # Perform set difference operation
    diff_np = np.setdiff1d(b_np, a_np)
    
    # Convert the result back to a PyTorch tensor
    diff_tensor = torch.tensor(diff_np, dtype=a.dtype)
    
    return diff_tensor

def get_svm_pred_proba(clf, embedding):
    """
    Given an SVM model and a 2D PyTorch tensor of embeddings, this function returns
    the predicted probabilities using the model's `predict_proba` method.
    
    Parameters:
    - clf : An SVM classifier model with Calibration
    - embedding (torch.Tensor): A 2D PyTorch tensor containing embeddings.
    
    Returns:
    - torch.Tensor: A 2D PyTorch tensor containing the predicted probabilities.
    """
    
    # Convert the PyTorch tensor to a NumPy array
    embedding_np = embedding.cpu().detach().numpy()
    
    # Use the SVM model to get predicted probabilities
    pred_proba_np = clf.predict_proba(embedding_np)
    
    # Convert the NumPy array back to a PyTorch tensor
    pred_proba_tensor = torch.tensor(pred_proba_np, dtype=torch.float32)
    
    return pred_proba_tensor


def reweight_valid_acc(valid_acc, temperature=1.0):
    """
    Reweights the validation accuracies using softmax with a temperature parameter.

    Parameters:
    - valid_acc (torch.Tensor): 1D tensor containing validation accuracies.
    - temperature (float): Temperature parameter for softmax.

    Returns:
    - reweighted_acc (torch.Tensor): 1D tensor containing reweighted validation accuracies.
    """
    valid_acc = valid_acc.view(-1,)
    valid_acc = -1.*valid_acc / temperature  # Apply temperature
    reweighted_acc = F.softmax(valid_acc, dim=0)  # Compute softmax along the dimension 0
    
    return reweighted_acc

def reweight_train_valid_acc_diff(train_valid_acc, temperature=1.0):
    """
    Reweights the differences between training and validation accuracies using softmax with a temperature parameter.

    Parameters:
    - train_valid_acc (list of tuples): List of tuples, each containing (train_acc, valid_acc).
    - temperature (float): Temperature parameter for softmax.

    Returns:
    - reweighted_acc_diff (torch.Tensor): 1D tensor containing reweighted differences.
    """
    # Calculate train_acc - valid_acc - 0.25 * valid_acc for each tuple
    acc_diff = torch.tensor([train_acc - valid_acc - 0.1 * valid_acc for train_acc, valid_acc in train_valid_acc])

    # Apply temperature scaling
    acc_diff_scaled = acc_diff / temperature

    # Compute softmax along the dimension 0
    reweighted_acc_diff = F.softmax(acc_diff_scaled, dim=0)

    return reweighted_acc_diff

def tensor_union(a, b):
    # Concatenate the two tensors
    concatenated = torch.cat((a, b))
    
    # Return the unique elements
    return torch.unique(concatenated)


def softmax_with_temperature_v2(T, temperature=1.0):
    """
    Calculate the softmax of a tensor with temperature.

    Parameters:
    T (torch.Tensor): A PyTorch tensor of shape (N, K).
    temperature (float): The temperature parameter for softmax.

    Returns:
    torch.Tensor: A tensor of shape (N, K) after applying softmax with temperature.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    # Adjust the tensor by the temperature
    T_adjusted = T / temperature
    # Apply softmax
    softmax_result = F.softmax(T_adjusted, dim=-1)
    return softmax_result

def softmax_with_temperature(values, temperature):
    """
    Apply softmax function to a list of values with a specified temperature.

    Parameters:
    values (list of float): The input values.
    temperature (float): The temperature parameter for softmax.

    Returns:
    list: Softmax-normalized values.
    """
    
    # Adjust values by temperature
    temp_values = np.array(values) / temperature
    # Compute softmax
    exp_values = np.exp(temp_values - np.max(temp_values))  # Subtract max for numerical stability
    return exp_values / exp_values.sum()


def reweight_sample(tensors, clfs, temp=1.0,reverse=False):
    """
    Processes a list of PyTorch tensors through multiple SVM classifiers to calculate entropy-based values.

    Parameters:
    tensors (list of torch.Tensor): A list of tensors, each corresponding to a classifier in 'clfs'.
    clfs (list of sklearn.svm.SVC): A list of trained SVM classifiers with probability prediction capability.
    sqrt (bool): If True, apply square root to the final result.

    Returns:
    dict: A dictionary where keys are indices and values are arrays of calculated values, one for each classifier.
    """
    if len(tensors) != len(clfs):
        raise ValueError("The number of tensors must match the number of classifiers.")

    result_dict = {idx: [] for idx in range(len(tensors[0]))}

    for tensor, clf in zip(tensors, clfs):
        # Convert tensor to NumPy array
        np_array = tensor.detach().cpu().numpy()

        # Get probability predictions for the current classifier
        prob_predictions = clf.predict_proba(np_array)

        # Calculate entropy
        entropy_values = entropy(prob_predictions.T, base=2)

        # Apply transformation
        transformed_values = entropy_values + 1e-8
        transformed_values = -1*transformed_values if reverse else transformed_values

        # Store the results in the dictionary
        for idx, value in enumerate(transformed_values):
            result_dict[idx].append(value)
    
    new_result_dict = {key: softmax_with_temperature(value, temp) for key, value in result_dict.items()}
    # for k in result_dict:
    #     print ('unnorm:',result_dict[k])
    #     print ("norm:",new_result_dict[k])
    #     if np.random.uniform()<0.1:
    #         break
    return new_result_dict

def dict_label_removed(tensor_dict, indices):
    """
    Indexes into the tensors of a dictionary with a set of indices to create sub-tensors.

    Parameters:
    tensor_dict (dict): A dictionary where each key corresponds to a tensor.
    indices (list or tensor): The indices to use for selecting the submatrix from each tensor.

    Returns:
    dict: A new dictionary with the same keys and sub-tensors as values.
    """
    indexed_dict = {key: tensor[indices] for key, tensor in tensor_dict.items()}
    return indexed_dict


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def visualize_logits_single_class(logits, Y, high_corrs, pos=0, dir='plots',dir_data=None, fname='plot'):
    """
    Visualize logits divided into groups based on high_corrs using Seaborn and save the plot.

    Args:
    - logits (numpy.ndarray): Logits array of shape (N, 3).
    - Y (numpy.ndarray): Labels array of shape (N,).
    - high_corrs (numpy.ndarray): High correlation array of shape (N,).
    - pos (int): Positive label indicator (default is 0).
    - dir (str): Directory to save the plot.
    - fname (str): File name for the saved plot.

    Returns:
    - None: This function saves the plot as a PDF.
    """

    # Identify correct positive predictions
    # correct_positives = (np.argmax(logits, axis=1) == Y) & (Y == pos)
    correct_positives = Y == pos
    # Filter logits for correct positive predictions
    # filtered_logits = logits[correct_positives, pos]
    filtered_logits = logits[correct_positives, pos]

    # Create a DataFrame for easier plotting
    data = pd.DataFrame({
        'Logit Values': filtered_logits,
        'High Correlation': high_corrs[correct_positives]
    })

    # Convert 'High Correlation' to categorical for better plotting
    data['High Correlation'] = data['High Correlation'].map({1: 'High (1)', 0: 'Low (0)'})

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create the box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='High Correlation', y='Logit Values', data=data)
    plt.title('Box Plot of Logits for High and Low Correlation Groups')
    plt.grid(True)

    # Check if directory exists, if not, create it
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)

    # Save the plot
    file_path = os.path.join(dir, f"{fname}.pdf")
    plt.savefig(file_path)
    print(f"Plot saved as {file_path}")
    
    # Save the DataFrame
    data_file_path = os.path.join(dir_data, f"{fname}.csv")
    data.to_csv(data_file_path, index=False)
    print(f"Data saved as {data_file_path}")
    
    #! get distribution for each class
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # for pos in range(3):  # Assuming classes are 0, 1, and 2
    #     # Identify predictions for the current class
    #     correct_predictions = Y == pos

    #     # Filter logits for the current class
    #     filtered_logits = logits[correct_predictions, pos]

    #     # Plot the histogram of filtered logits for the current class
    #     axs[pos].hist(filtered_logits, bins=30, alpha=0.75, color='blue')
    #     axs[pos].set_title(f'Class {pos} Logits')
    #     axs[pos].set_xlabel('Logit Value')
    #     axs[pos].set_ylabel('Frequency')
    # plt.tight_layout()
    # plot_path = os.path.join(dir, 'allLabels_distributions.pdf')
    # plt.savefig(plot_path)
    # plt.close()



def visualize_logits_all_class(logits, Y, high_corrs, dir='plots', fname='plot', dir_data='data'):
    """
    Visualize logits for each class divided into groups based on high_corrs using Seaborn, 
    save the plot, and save the DataFrame as a CSV file.
    
    Args:
    - logits (numpy.ndarray): Logits array of shape (N, 3).
    - Y (numpy.ndarray): Labels array of shape (N,).
    - high_corrs (numpy.ndarray): High correlation array of shape (N,).
    - dir (str): Directory to save the plot.
    - fname (str): File name for the saved plot.
    - dir2 (str): Directory to save the DataFrame.

    Returns:
    - None: This function saves the plot as a PDF and the DataFrame as a CSV.
    """

    all_data = []

    # Process logits for each class
    for pos in range(3):
        # Identify correct positive predictions for each class
        # correct_positives = (np.argmax(logits, axis=1) == Y) & (Y == pos)
        correct_positives = Y == pos

        # Filter logits for correct positive predictions
        # filtered_logits = logits[correct_positives, pos]
        filtered_logits = logits[correct_positives, pos]

        # Create a DataFrame for each class
        data = pd.DataFrame({
            'Logit Values': filtered_logits,
            'High Correlation': high_corrs[correct_positives],
            'Class': np.full(filtered_logits.shape, pos)  # Add class label
        })

        all_data.append(data)

    # Concatenate data for all classes
    concatenated_data = pd.concat(all_data)

    # Convert 'High Correlation' and 'Class' to categorical for better plotting
    concatenated_data['High Correlation'] = concatenated_data['High Correlation'].map({1: 'High (1)', 0: 'Low (0)'})
    concatenated_data['Class'] = concatenated_data['Class'].map({0: 'Class 0', 1: 'Class 1', 2: 'Class 2'})

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create the box plot
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Class', y='Logit Values', hue='High Correlation', data=concatenated_data)
    # plt.title('Box Plot of Logits for High and Low Correlation Groups Across Classes')
    plt.grid(True)

    # Check if plot directory exists, if not, create it
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)

    # Check if plot file exists, if so, delete it
    plot_file_path = os.path.join(dir, f"{fname}.pdf")
    if os.path.exists(plot_file_path):
        os.remove(plot_file_path)

    # Save the plot
    plt.savefig(plot_file_path)
    print(f"Plot saved as {plot_file_path}")

    # Check if data directory exists, if not, create it
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)

    # Save the DataFrame
    data_file_path = os.path.join(dir_data, f"{fname}.csv")
    concatenated_data.to_csv(data_file_path, index=False)
    print(f"Data saved as {data_file_path}")

# Example usage
# logits = np.random.rand(100, 3)  # Example logits
# Y = np.random.randint(0, 3, 100)  # Example labels
# high_corrs = np.random.randint(0, 2, 100)  # Example high_corrs
# visualize_corrected_logits_seaborn(logits, Y, high_corrs, dir='my_plots', fname='my_plot')


def plot_umap_embeddings(emb, label, save_path):
    """
    Reduces the dimensionality of embeddings and plots them.

    Parameters:
    emb (torch.Tensor): A PyTorch tensor of shape (N, 32).
    label (torch.Tensor): A PyTorch tensor of shape (N,), with values 0 or 1.
    save_path (str): File path to save the plot as a PDF.

    The function converts the tensors to NumPy format, applies UMAP to reduce
    the dimensionality to 2D, plots the embeddings with different colors based
    on the label, and saves the plot to the specified path.
    """

    # Convert PyTorch tensors to NumPy arrays
    emb_np = emb.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()

    # Dimensionality reduction using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_reduced = reducer.fit_transform(emb_np)
    sns.set(style="darkgrid")
    # Plotting
    plt.figure(figsize=(10, 8))
    for i in range(2):
        indices = label_np == i
        plt.scatter(emb_reduced[indices, 0], emb_reduced[indices, 1], 
                    c=('blue' if i == 0 else 'red'), label=str(i), alpha=0.5)

    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf')
    plt.close()



def plot_classes(group0_class0, group1_class0, group0_class1, group1_class1, group0_class2, group1_class2,title='try',save_path='plots/try.pdf'):
    # Set the aesthetic style of the plots
    sns.set()

    # Colors and markers for visualization
    colors = ['red', 'blue', 'black']  # Base colors for classes
    markers = ['o', '^']  # Circle for group 0, Triangle for group 1
    alphas = [1.0, 0.5]  # Alpha values for transparency

    # Create a scatter plot
    plt.figure(figsize=(8, 6))

    # Plotting each group of each class
    for i, class_data in enumerate([(group0_class0, group1_class0), 
                                    (group0_class1, group1_class1), 
                                    (group0_class2, group1_class2)]):
        for j, group_data in enumerate(class_data):
            sns.scatterplot(x=group_data[:, 0], y=group_data[:, 1], 
                            label=f'Class {i} Group {j}', 
                            color=colors[i], 
                            marker=markers[j],
                            alpha=alphas[j])

    plt.title(title)
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
    plt.savefig(save_path)


def cmd(x1, x2, n_moments=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = dm
    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms+=moment_diff(sx1,sx2,i+2)
    return scms
    
def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    val =  ((x1-x2)**2).sum()
    return np.sqrt(val)

def moment_diff(sx1, sx2, k):
    """
    Difference between the k-th moments of sx1 and sx2.
    """
    # Calculate the k-th moment for each set of samples
    ss1 = np.mean(sx1**k, axis=0)
    ss2 = np.mean(sx2**k, axis=0)

    # Calculate the L2 norm (Euclidean distance) between the two moments
    return l2diff(ss1, ss2)


import numpy as np
from sklearn.manifold import TSNE

def apply_tsne_and_split(arrays, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    """
    Apply t-SNE to concatenated arrays and split back according to original shapes.

    Parameters:
    arrays (list of numpy.ndarray): List of arrays to concatenate and transform.
    n_components, perplexity, learning_rate, n_iter: t-SNE parameters.

    Returns:
    list of numpy.ndarray: Transformed arrays split according to original shapes.
    """
    # Concatenate the arrays
    concatenated_array = np.concatenate(arrays, axis=0)

    # Apply t-SNE
    tsne = TSNE(n_components=n_components)
    transformed_data = tsne.fit_transform(concatenated_array)

    # Split the transformed data back into individual arrays
    split_indices = np.cumsum([arr.shape[0] for arr in arrays[:-1]])
    split_arrays = np.split(transformed_data, split_indices)

    return split_arrays

import re

def extract_hidden_dims(input_string):
    """
    Extracts the number following 'hidden_dims_' in the input string.

    Args:
    input_string (str): The input string containing the pattern.

    Returns:
    int: The extracted number or None if the pattern is not found.
    """
    match = re.search(r'hidden_dims_(\d+)', input_string)
    if match:
        return int(match.group(1))
    else:
        return None


def calculate_entropies(clf, emb):
    # Check if emb is a PyTorch tensor and convert it to a NumPy array
    if isinstance(emb, torch.Tensor):
        emb_np = emb.detach().cpu().numpy()
    else:
        raise TypeError("emb must be a PyTorch tensor")

    # Ensure the shape of the array is (N, D)
    if emb_np.ndim != 2:
        raise ValueError("emb must be a 2-dimensional array")

    # Predict probabilities
    probabilities = clf.predict_proba(emb_np)

    # Calculate entropy for each sample
    entropies = np.apply_along_axis(entropy, 1, probabilities)

    return entropies


if __name__ == '__main__':
    # Example usage
    N, D = 100, 20  # Number of samples and dimensions
    K = 10  # Number of top-K similar samples to consider
    X = torch.rand(N, D)  # Random tensor of shape (N, D)
    Y = torch.randint(0, 2, (N,))  # Random tensor of shape N with values in {0, 1}

    average_score = compute_average_score(X, Y, K)
    print("Average Score:", average_score)





