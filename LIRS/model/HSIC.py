import torch
import torch.nn as nn
class HSIC(nn.Module):
    def __init__(self, kernel_method='rbf', sigma=1.0,device=0):
        super(HSIC, self).__init__()
        self.kernel_method = kernel_method
        self.sigma = sigma
        self.device = device

    @staticmethod
    def compute_centering_matrix(n):
        I = torch.eye(n)
        ones = torch.ones((n, n))
        C = I - ones / n
        return C

    def compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        
        if self.kernel_method == 'linear':
            K = torch.matmul(X, Y.T)
        elif self.kernel_method == 'rbf':
            X_norm = (X ** 2).sum(1).view(-1, 1)
            Y_norm = (Y ** 2).sum(1).view(1, -1)
            K = torch.exp(-0.5 * (X_norm + Y_norm - 2 * torch.matmul(X, Y.T)) / (self.sigma ** 2))
        else:
            raise ValueError("Unsupported kernel method. Choose 'rbf' or 'linear'.")
    
        return K

    def compute_hsic(self, X, Y):
        n = X.shape[0]
        
        # Compute the centering matrix
        C = self.compute_centering_matrix(n)
        C = C.to(self.device)
        
        # Compute the kernel matrices for X and Y
        K_X = self.compute_kernel(X)
        K_Y = self.compute_kernel(Y)
        
        # Compute the HSIC value
        hsic_value = (1 / (n - 1) ** 2) * torch.trace(torch.matmul(torch.matmul(K_X, C), torch.matmul(K_Y, C)))
        
        return hsic_value



if __name__ == '__main__':
    # Example usage:
    N, F = 30, 10
    X = torch.randn(N, F)
    Y = torch.randn(N, F)

    hsic_rbf = HSIC(kernel_method='rbf', sigma=0.1)
    hsic_value_rbf = hsic_rbf.compute_hsic(X, Y)
    print(f"HSIC value (RBF kernel): {hsic_value_rbf.item()}")

    hsic_linear = HSIC(kernel_method='linear')
    hsic_value_linear = hsic_linear.compute_hsic(X, Y)
    print(f"HSIC value (linear kernel): {hsic_value_linear.item()}")
