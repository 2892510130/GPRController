from train import GPy_test, Sparse_GPy_test, write_train_data

def test_train_gp():
    write_data = False
    sparse = True
    plot_enable = True

    file_name = "test.txt"
    log_dir = "log/gpr_log/" + file_name

    p = {
        "train_data_number": 4000, 
        "test_data_number": 800, 
        "input_window": 3, 
        "gap_window": 0,
        "train_file_path": log_dir,
        "test_file_path": log_dir,
        "fixed_data_path": f"log/processed_data/{file_name}",
        "num_inducing": 40
    }

    test = Sparse_GPy_test if sparse else GPy_test

    if write_data:
        write_train_data(p)
        
    test(p, print_enable=True, plot_enable=plot_enable, optimize=True, write_data=write_data, output_id=0)
    test(p, print_enable=True, plot_enable=plot_enable, optimize=True, write_data=write_data, output_id=1)