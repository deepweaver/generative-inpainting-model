wget -P ./checkpoints/ http://35.225.2.172/imagenet.tar 
# wget -P ./checkpoints/ https://storage.cloud.google.com/buttcket/imagenet.tar
tar xvf ./checkpoints/imagenet.tar 
wget -P ./raven_data/ http://35.225.2.172/data_a_ab_c_d_e_with_mask_570x900_py3.pkl
# wget -P ./raven_data/ https://storage.cloud.google.com/buttcket/data_a_ab_c_d_e_with_mask_570x900_py3.pkl
wget -P ./data/datasets/ https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -C data/datasets/ -xzvf data/datasets/dtd-r1.0.1.tar.gz
# we have the dtd/ folder images in ./data/datasets/dtd/images/






