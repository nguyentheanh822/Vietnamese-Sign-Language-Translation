with open("source/train.py", "r") as f:
    lines = f.readlines()

with open("source/train.py", "w") as f:
    for line in lines:
        if "device = torch.device('cuda:0'" in line:
            f.write("    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n")
        else:
            f.write(line)
