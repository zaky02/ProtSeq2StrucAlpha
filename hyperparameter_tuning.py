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
    "test_split": 0.20,
    "masking_ratio": 0.15,
    "epsilon": 1e-8,
    "verbose": 1,
    "max_len": 1024,
    "num_gpus": 4,
    "parallel_strategy": "ddp",
    "draw_model": False,
    "early_stopping_patience": 5,
    "early_stopping_delta": 0.001
}

lr = [0.00001, 0.0001, 0.001, 0.01]
weight_decay = [0.00001, 0.0001, 0.001, 0.01]
dropout = [0.1, 0.3, 0.5, 0.8]
batch_size = [64, 128, 256, 512, 1024]
dim_model = [256, 512, 1024, 2048]
ff_hidden_layer = [1024, 2048, 4096, 8192]
num_heads = [4, 8, 16, 32]
num_layers = [3, 6, 12, 24]

for i in lr:
    for j in weight_decay:
        for w in dropout:
            for z in batch_size:
                for c in dim_model:
                    for t in ff_hidden_layer:
                        for m in num_heads:
                            for n in num_layers:
                                if c % m != 0:
                                    continue
                                if t % c != 0:
                                    continue
                                if z % json_dictionary["num_gpus"] != 0:
                                    continue

                                config = json_dictionary.copy()
                                config.update({
                                    "learning_rate": i,
                                    "weight_decay": j,
                                    "dropout": w,
                                    "batch_size": z,
                                    "dim_model": c,
                                    "ff_hidden_layer": t,
                                    "num_heads": m,
                                    "num_layers": n
                                })

                                filename = f"config_lr{i}_wd{j}_do{w}_bs{z}_dm{c}_ff{t}_nh{m}_nl{n}"
                                filename = filename.replace('.', '')

                                with open(f"configs/{filename}.json", "w") as f:
                                    json.dump(config, f, indent=4)

