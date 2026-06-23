import torch

class StandardScaler():
    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.column_mask: torch.Tensor | None = None

    def fit_transform(self, 
                      X: torch.Tensor, 
                      column_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)
        self.std = torch.where(self.std < 1e-15, 1.0, self.std) # do not scale where standard deviation is zero
        self.column_mask = (
            torch.ones_like(self.mean, dtype=torch.bool) 
            if column_mask is None else column_mask
        )

        return self.transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None or self.column_mask is None:
            raise TypeError("Mean and standard deviation must be learnt. You might want to use `fit_transform` first.")
        
        # This assertion block is just for Pylance
        assert isinstance(self.mean, torch.Tensor), ""
        assert isinstance(self.std, torch.Tensor)
        assert isinstance(self.column_mask, torch.Tensor)

        masked_mean: torch.Tensor = self.column_mask * self.mean
        masked_std: torch.Tensor = torch.exp(self.column_mask * torch.log(self.std))
        return (X - masked_mean) / masked_std
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)
    

if __name__ == '__main__':
    x = torch.tensor(
        [[0, 1., 0, 5.],
         [0, 2., 1, 25.],
         [1, 3., 1, 125.],
         [1, 4., 0, 625.],
         [1, 5., 0, 625.]]
    )
    y = torch.tensor([[0, 6., 1, 1.]])
    scaler = StandardScaler()
    mask = torch.tensor([False, True, False, True])
    scaler.fit_transform(x, mask)
    x_scaled = scaler(x)
    print(x)
    print(x_scaled)
    y_scaled = scaler(y)
    print(y)
    print(y_scaled)
