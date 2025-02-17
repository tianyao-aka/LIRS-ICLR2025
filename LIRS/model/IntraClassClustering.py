import torch
import numpy as np
from sklearn.cluster import KMeans

class ClusterProcessor:
    def __init__(self, spu_emb, labels, num_classes):
        """
        Initialize the ClusterProcessor with embeddings, labels, and number of classes.

        Args:
            spu_emb (torch.Tensor): A tensor of shape (N, D) representing the embeddings.
            labels (torch.Tensor): A tensor of shape (N,) representing the class labels.
            num_classes (int): The number of classes.
        """
        self.spu_emb = spu_emb.cpu().detach().numpy()
        self.labels = labels.cpu().detach().numpy()
        self.num_classes = num_classes
        self.binary_cid = None
        self.cluster_info = None

    def process(self):
        """
        Process the embeddings and labels to perform KMeans clustering for each class.
        Save binary cluster IDs and cluster information.
        """
        N = self.spu_emb.shape[0]
        binary_cid = torch.zeros(N, dtype=torch.int)
        cluster_info = [(0, 0)] * N

        for i in range(self.num_classes):
            mask = self.labels == i
            spu_emb_class = self.spu_emb[mask]
            indices = np.where(mask)[0]

            if spu_emb_class.shape[0] > 1:
                kmeans = KMeans(n_clusters=2, random_state=1)
                kmeans.fit(spu_emb_class)
                cluster_labels = kmeans.labels_

                # Calculate the number of samples in each cluster
                unique, counts = np.unique(cluster_labels, return_counts=True)
                num_samples_per_cluster = dict(zip(unique, counts))

                # Identify the largest cluster
                largest_cluster_id = max(num_samples_per_cluster, key=num_samples_per_cluster.get)
                largest_cluster_mask = cluster_labels == largest_cluster_id

                for idx, is_in_largest in zip(indices, largest_cluster_mask):
                    binary_cid[idx] = int(is_in_largest)
                    cluster_info[idx] = (np.sum(largest_cluster_mask), spu_emb_class.shape[0] - np.sum(largest_cluster_mask))

        self.binary_cid = binary_cid
        self.cluster_info = cluster_info
        return self.binary_cid,self.cluster_info

    def get_binary_cid(self):
        """
        Return the list of binary cluster IDs.

        Returns:
            list: A list of binary cluster IDs for each sample.
        """
        if self.binary_cid is None:
            raise ValueError("You need to call process() before getting the binary cluster IDs.")
        return self.binary_cid.tolist()

    def get_cluster_info(self):
        """
        Return the list of tuples of (samples_in_cluster, total_samples).

        Returns:
            list: A list of tuples containing the number of samples in the largest cluster and total samples for each sample.
        """
        if self.cluster_info is None:
            raise ValueError("You need to call process() before getting the cluster info.")
        return self.cluster_info
    

