import torch
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import time


class ClusterProcessor:
    def __init__(self, spu_emb, labels, num_classes, num_clusters, gamma=1.0):
        """
        Initialize the ClusterProcessor with embeddings, labels, number of classes, and number of clusters.

        Args:
            spu_emb (torch.Tensor): A tensor of shape (N, D) representing the embeddings.
            labels (torch.Tensor): A tensor of shape (N,) representing the class labels.
            num_classes (int): The number of classes.
            num_clusters (int): The number of clusters for KMeans.
            gamma (float): The parameter for the reweighting function.
        """
        self.spu_emb = spu_emb.cpu().detach().numpy()
        self.labels = labels.cpu().detach().numpy()
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.binary_cid = None
        self.cluster_info = None
        self.logits = None
        self.cluster_labels = None
        self.sample_weights = None

    def process(self):
        """
        Process the embeddings and labels to perform KMeans clustering for each class.
        Save binary cluster IDs, cluster information, compute logits using LinearSVC with probability calibration, 
        store cluster labels, and compute sample weights.

        Returns:
            torch.Tensor: A tensor of shape (N, num_clusters) representing the logits for each sample.
            torch.Tensor: A tensor of shape (N,) representing the cluster labels for each sample.
            torch.Tensor: A tensor of shape (N,) representing the sample weights for each sample.
        """
        
        N = self.spu_emb.shape[0]
        binary_cid = torch.zeros(N, dtype=torch.int)
        cluster_info = [(0, 0)] * N
        logits = np.zeros((N, self.num_clusters))
        cluster_labels = np.zeros(N, dtype=int)
        sample_weights = np.zeros(N, dtype=float)

        for i in range(self.num_classes):
            mask = self.labels == i
            spu_emb_class = self.spu_emb[mask]
            indices = np.where(mask)[0]

            if spu_emb_class.shape[0] > 1:
                # kmeans = KMeans(n_clusters=self.num_clusters, random_state=1)
                print ('start fitting mini batch kmeans')
                s = time.time()
                kmeans = MiniBatchKMeans(n_clusters=self.num_clusters,max_iter=140,batch_size=1024)
                kmeans.fit(spu_emb_class)
                class_cluster_labels = kmeans.labels_
                t = time.time()
                print (f"end fitting minibatch kmeans. label:{i}, running time:{t-s}")
                # Store the cluster labels
                for idx, label in zip(indices, class_cluster_labels):
                    cluster_labels[idx] = label
                
                # Calculate cluster counts and normalize
                unique, counts = np.unique(class_cluster_labels, return_counts=True)
                probs = counts / counts.sum()
                print (f'class label:{self.num_classes}. cluster_num counts:{counts}')
                # Apply the reweighting function
                if self.gamma>0:
                    weights = (1. - probs ** self.gamma) / self.gamma
                else:
                    probs_ = probs*0. + 1/self.num_clusters
                    weights = probs_

                # Assign weights to samples
                for idx, label in zip(indices, class_cluster_labels):
                    sample_weights[idx] = weights[label]

                # Train LinearSVC and calibrate probabilities
                svc = LinearSVC()
                calibrated_svc = CalibratedClassifierCV(svc,cv=3)
                calibrated_svc.fit(spu_emb_class, class_cluster_labels)
                class_logits = calibrated_svc.predict_proba(spu_emb_class)
                class_predictions = calibrated_svc.predict(spu_emb_class)
                acc = accuracy_score(class_cluster_labels, class_predictions)
                print (f"svm fitting cluster labels. label:{i}, Accuracy:{acc}")
                # Store logits
                for idx, logit in zip(indices, class_logits):
                    logits[idx, :] = logit

        self.binary_cid = binary_cid
        self.cluster_info = cluster_info
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.cluster_labels = torch.tensor(cluster_labels, dtype=torch.int)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        return self.logits, self.cluster_labels, self.sample_weights


    def processV2(self):
        """
        Process the embeddings and labels to perform KMeans clustering for each class.
        Save binary cluster IDs, cluster information, compute logits using LinearSVC with probability calibration, 
        store cluster labels, and compute sample weights based on class logits and the majority cluster.

        Returns:
            torch.Tensor: A tensor of shape (N, num_clusters) representing the logits for each sample.
            torch.Tensor: A tensor of shape (N,) representing the cluster labels for each sample.
            torch.Tensor: A tensor of shape (N,) representing the sample weights for each sample.
        """
        N = self.spu_emb.shape[0]
        binary_cid = torch.zeros(N, dtype=torch.int)
        cluster_info = [(0, 0)] * N
        logits = np.zeros((N, self.num_clusters))
        cluster_labels = np.zeros(N, dtype=int)
        sample_weights = np.zeros(N, dtype=float)

        for i in range(self.num_classes):
            mask = self.labels == i
            spu_emb_class = self.spu_emb[mask]
            indices = np.where(mask)[0]

            if spu_emb_class.shape[0] > 1:
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=1)
                kmeans.fit(spu_emb_class)
                class_cluster_labels = kmeans.labels_

                # Store the cluster labels
                for idx, label in zip(indices, class_cluster_labels):
                    cluster_labels[idx] = label

                # Calculate cluster counts and normalize
                unique, counts = np.unique(class_cluster_labels, return_counts=True)
                probs = counts / counts.sum()

                # Find the label of the largest cluster
                majority_cluster_label = unique[np.argmax(counts)]

                # Train LinearSVC and calibrate probabilities
                svc = LinearSVC()
                calibrated_svc = CalibratedClassifierCV(svc)
                calibrated_svc.fit(spu_emb_class, class_cluster_labels)
                class_logits = calibrated_svc.predict_proba(spu_emb_class)
                class_predictions = calibrated_svc.predict(spu_emb_class)
                acc = accuracy_score(class_cluster_labels, class_predictions)
                print(f"svm fitting cluster labels. label:{i}, Accuracy:{acc}")

                # Apply the reweighting function using logits of the majority cluster
                majority_cluster_probs = class_logits[:, majority_cluster_label]
                weights = (1. - majority_cluster_probs ** self.gamma) / self.gamma

                # Assign weights to samples
                for idx, weight in zip(indices, weights):
                    sample_weights[idx] = weight

                # Store logits
                for idx, logit in zip(indices, class_logits):
                    logits[idx, :] = logit

        self.binary_cid = binary_cid
        self.cluster_info = cluster_info
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.cluster_labels = torch.tensor(cluster_labels, dtype=torch.int)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        return self.logits, self.cluster_labels, self.sample_weights

    def process_no_cluster(self):
        """
        Process the embeddings and labels without clustering, using LinearSVC with probability calibration.
        Calculate sample weights based on the probability score of the correct class label.

        Returns:
            torch.Tensor: A tensor of shape (N, num_classes) representing the logits for each sample.
            torch.Tensor: A tensor of shape (N,) representing the class labels for each sample.
            torch.Tensor: A tensor of shape (N,) representing the sample weights for each sample.
        """
        N = self.spu_emb.shape[0]
        logits = np.zeros((N, self.num_classes))
        sample_weights = np.zeros(N, dtype=float)

        # Train LinearSVC and calibrate probabilities
        svc = LinearSVC()
        calibrated_svc = CalibratedClassifierCV(svc)
        calibrated_svc.fit(self.spu_emb, self.labels)
        pred_proba = calibrated_svc.predict_proba(self.spu_emb)

        # Calculate sample weights
        for i in range(N):
            correct_class_prob = pred_proba[i, int(self.labels[i])]
            sample_weights[i] = (1. - correct_class_prob ** self.gamma) / self.gamma

        self.logits = torch.tensor(pred_proba, dtype=torch.float32)
        self.cluster_labels = torch.tensor(self.labels, dtype=torch.int)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        return self.logits, self.cluster_labels, self.sample_weights
    
    