with open("source/train.py", "r") as f:
    lines = f.readlines()

with open("source/train.py", "w") as f:
    for line in lines:
        if "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")" in line:
            # force to cuda:0 since CUDA_VISIBLE_DEVICES shields other GPUs anyway
            f.write("    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n")
        else:
            f.write(line)
