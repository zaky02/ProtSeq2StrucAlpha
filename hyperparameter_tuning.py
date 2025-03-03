import json
import os

os.makedirs("configs", exist_ok=True)

json_dictionary = {
    "data_as_csv": "structures/mane_overlap_v4_filt.csv",
    "data_as_pdbs": "structures/MANE_overlap_filt",
    "foldseek_path": "bin/foldseek",
    "weight_path": "model_weights.pth",
    "get_wandb": True,
    "wandb_project": "ProtSeq2StrucAlpha_v0",
    "epochs": 10,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 64,
    "test_split": 0.20,
    "masking_ratio": 0.15,
    "epsilon": 1e-8,
    "verbose": 1,
    "dropout": 0.1,
    "max_len": 1024,
    "num_gpus": 4,
    "parallel_strategy": "ddp",
    "draw_model": False,
    "early_stopping_patience": 15,
    "early_stopping_delta": 0.001
}


dim_model = [256, 512, 1024, 2048]
ff_hidden_layer = [1024, 2048, 4096, 8192]
num_heads = [4, 8, 16, 32]
num_layers = [3, 6, 12, 24]

for c in dim_model:
    for m in num_heads:
        for n in num_layers:
            for t in ff_hidden_layer:
                if c % m != 0:
                    continue
                if t != 4*c:
                    continue

                config = json_dictionary.copy()
                config.update({"dim_model": c,
                               "num_heads": m,
                               "num_layers": n,
                               "ff_hidden_layer": t})
                    
                filename = f"config_dm{c}_nh{m}_nl{n}_ff{t}"
                filename = filename.replace('.', '')

                with open(f"configs/{filename}.json", "w") as f:
                    json.dump(config, f, indent=4)

