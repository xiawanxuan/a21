import torch


class Config:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data"
    model_dir = "./models"
    result_dir = "./results"
    output_dir = "./outputs"

    spectrum_length = 1024
    num_classes = 5
    class_names = ["Galaxy", "Quasar", "Star", "Nebula", "Unknown"]

    d_model = 128
    nhead = 4
    num_encoder_layers = 4
    dim_feedforward = 256
    dropout = 0.1
    activation = "gelu"

    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    warmup_epochs = 5

    red_shift_loss_weight = 1.0
    classification_loss_weight = 1.0

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    noise_level = 0.05
    augment_prob = 0.5

    save_interval = 5
    log_interval = 10
