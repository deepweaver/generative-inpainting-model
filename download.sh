wget -P ./checkpoints/ http://35.238.195.83/imagenet.tar 
tar xvf ./checkpoints/imagenet.tar 
wget -P ./raven_data/ http://35.238.195.83/data_a_ab_c_d_e_with_mask_570x900_py3.pkl

wget -P ./data/datasets/ https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -C data/datasets/ -xzvf data/datasets/dtd-r1.0.1.tar.gz
# we have the dtd/ folder images in ./data/datasets/dtd/images/






