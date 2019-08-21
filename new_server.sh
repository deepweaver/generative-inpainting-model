sudo apt-get install zsh 
sh -c "$(wget -O- https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
echo "zsh" >> ~/.bashrc
# source ~/.bashrc

wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
cd ~
sh ./Miniconda3-latest-Linux-x86_64.sh
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc 
source ~/.zshrc 
cd generative-inpainting-model/
pip install -r requirements.txt

sh ./download.sh 

echo "You have successfully completed installation and downloading" 







