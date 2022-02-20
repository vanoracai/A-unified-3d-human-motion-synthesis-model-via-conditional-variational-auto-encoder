# download hm36 data
cd data/hm36
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1P7W3ldx2lxaYJJYcf3RG4Y9PsD4EJ6b0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1P7W3ldx2lxaYJJYcf3RG4Y9PsD4EJ6b0" -O data_3d_h36m.npz && rm -rf /tmp/cookies.txt

# single-gpu: train on human 3.6M without action label
python train_pose.py --config ./config/hm36/non_action_hm36.yaml

# single-gpu: train on human 3.6M with action label 
python train_pose.py --config ./config/hm36/w_action_hm36.yaml

