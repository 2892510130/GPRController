from .noise_model import GaussianNoiseModel
from .discrete import DiscreteMethods
from .cartesian_model import CartesianModel, CartesianDelayModel
from .frenet_model import FrenetModel
from .cartesian_gp_model import CartesianGaussianModel, AugmentedCartesianGaussianModel
from .frenet_gp_model import FrenetGaussianModel, AugmentedFrenetGaussianModel

ModelLists = {
    "Kinematic Cartesian":                    CartesianModel,
    "Kinematic Cartesian Delay":              CartesianDelayModel,
    "Kinematic Cartesian Gaussian":           CartesianGaussianModel,
    "Kinematic Augmented Cartesian Gaussian": AugmentedCartesianGaussianModel,
    "Kinematic Frenet":                       FrenetModel,
    "Kinematic Frenet Gaussian":              FrenetGaussianModel,
    "Kinematic Augmented Frenet Gaussian":    AugmentedFrenetGaussianModel,
}