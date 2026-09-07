from utils import Config

def test_gp_model_error():
    config = Config.from_yaml("config/mpc/linear_frenet_modest.yaml") # linear_frenet_modest linear_mpc_frenet
    gp_config = config.gaussian
    from model.gp_model import GaussianModel, test_gaussian_model
    print(gp_config.path, gp_config.total_data_number)
    model = GaussianModel(gp_config)
    test_gaussian_model(model)

    gp_config.sparse_method = 2
    model = GaussianModel(gp_config)
    test_gaussian_model(model)