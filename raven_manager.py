import os 

dirname = 'checkpoints/imagenet/hole_benchmark' 





gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and 'gen' in f and ".pt" in f]

gen_models.sort() 
print(gen_models) 












