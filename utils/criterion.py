import numpy as np

class Criterion():
    def __init__(self, method: str = "MSE") -> None:
        self.method = method
    
    def compute(self, case: str, **kwargs) -> np.ndarray:
        switch_dict = \
        {
            "MSE": self.mse_criterion,
            "RMSE": self.rmse_criterion,
            "SMSE": self.smse_criterion,
            "MAE": self.mae_criterion,
            "MRSE": self.mrse_criterion,
            "LPD": self.lpd_criterion,
            "MSLL": self.msll_criterion,
        }

        func = switch_dict.get(case, lambda **kwargs: "Error in the switch case.")
        return func(**kwargs)

    def mse_criterion(self, output :np.ndarray, target :np.ndarray) -> np.ndarray:
        return np.mean((output - target)**2)
    
    def rmse_criterion(self, output :np.ndarray, target :np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((output - target)**2))
    
    def smse_criterion(self, output :np.ndarray, target :np.ndarray) -> np.ndarray:
        return np.mean((output - target)**2) / np.var(output)
    
    def mae_criterion(self, output :np.ndarray, target :np.ndarray) -> np.ndarray:
        return np.abs(output - target).mean()
    
    def mrse_criterion(self, output :np.ndarray, target :np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((output - target)**2) / np.sum(target**2))
    
    def lpd_criterion(self, output :np.ndarray, target :np.ndarray, output_var :np.ndarray) -> np.ndarray:
        return 0.5 * np.log(2 * np.pi) + 0.5 * np.mean(np.log(output_var) + (output - target)**2 / output_var)
    
    def msll_criterion(self, output :np.ndarray, target :np.ndarray, output_var :np.ndarray) -> np.ndarray:
        target_mean = np.mean(target)
        target_var = np.var(target)
        return 0.5 * (np.mean(np.log(output_var) + (output - target)**2 / output_var
                       - (target - target_mean)**2 / target_var - np.log(target_var)))