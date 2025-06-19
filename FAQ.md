WSL环境下安装:
前置条件：bios启用虚拟化，win10以上系统，安装英伟达驱动，终端启用功能：
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
重新启动

终端运行：
wsl --install

配置用户，密码

启动ubuntu安装匹配cuda版本的gcc

终端运行：
sudo apt install gcc-11 g++-11
sudo ln -s /usr/bin/gcc-11 /usr/local/cuda/bin/gcc 
sudo ln -s /usr/bin/g++-1 /usr/local/cuda/bin/g++

获取并安装cudatoolkit （参考https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo env CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 sh cuda_11.8.0_520.61.05_linux.run

sudo ln -s /usr/bin/gcc-11 /usr/local/cuda/bin/gcc 
sudo ln -s /usr/bin/g++-11 /usr/local/cuda/bin/g++