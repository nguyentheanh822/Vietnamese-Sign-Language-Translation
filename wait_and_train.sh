#!/bin/bash
# wait_and_train.sh
# Script to wait for GPU 1 memory to free up (under 5GB) and then start the SOTA training

echo "Dang cho card GPU 1 trong..." >> results/run_SOTA_stgcn/train_output.txt

while true; do
    # Get memory used by GPU 1 in MiB
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    
    if [ "$MEM_USED" -lt 5000 ]; then
        echo "GPU 1 da san sang (Memory: ${MEM_USED} MiB). Bat dau train SOTA..." >> results/run_SOTA_stgcn/train_output.txt
        echo "=====================================" >> results/run_SOTA_stgcn/train_output.txt
        
        # Restore the original fix_device.py mapping (CUDA_VISIBLE_DEVICES=1 -> cuda:0 internally)
        cat << 'PYEOF' > fix_device_restore.py
with open("source/train.py", "r") as f:
    lines = f.readlines()

with open("source/train.py", "w") as f:
    for line in lines:
        if "device = torch.device('cuda:0'" in line:
            f.write("    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n")
        else:
            f.write(line)
PYEOF
        python3 fix_device_restore.py
        
        # Start training
        CUDA_VISIBLE_DEVICES=1 python3 source/train.py --config source/config/config_sota.yaml >> results/run_SOTA_stgcn/train_output.txt 2>&1
        break
    else:
        # Wait 5 minutes before checking again to avoid CPU spam
        sleep 300
    fi
done
