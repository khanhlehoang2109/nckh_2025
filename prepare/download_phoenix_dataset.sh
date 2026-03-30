#!/bin/bash
mkdir -p dataset/
cd dataset/

echo "The datasets will be stored in the 'dataset' folder\n"

# Phoenix sign language
echo "Downloading the Phoenix sign language dataset"
gdown "https://drive.google.com/uc?id=18w5ZkXGXY9EJRlleBaqodfIadWCdJ_5w"
echo "Extracting the Phoenix sign language dataset"
unzip diffusion_phoenix_dataset.zip
mv diffusion_phoenix_dataset PHOENIX
echo "Cleaning\n"
rm diffusion_phoenix_dataset.zip

echo "Downloading done!"
