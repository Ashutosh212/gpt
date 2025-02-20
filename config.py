from pathlib import Path

def get_config():
    return {
        "batch_size" : 2,
        "num_epochs":2,
        "lr": 10**-4,
        "seq_len" : 2705,
        "d_model" : 32,
        "lang_src" : "english",
        "lang_tgt" : "hindi",
        "datasource": 'opus_books',
        "model_folder" : "weights",
        "model_filename" : "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/model"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])