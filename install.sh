#!/bin/bash

# Get the directory where this script is located, and cd to it
THISDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variable
echo "Exporting INFERNUS_DIR=${THISDIR}"
export INFERNUS_DIR="${THISDIR}"

# Append to ~/.bashrc if not already present
if ! grep -q "export INFERNUS_DIR=" ~/.bashrc; then
	echo "Appending INFERNUS_DIR to ~/.bashrc"
	echo "export INFERNUS_DIR=\"${THISDIR}\"" >> ~/.bashrc
else
	echo "INFERNUS_DIR already set in ~/.bashrc. Overwriting..."
	#overwrite the existing line to ensure it points to the correct directory
	sed -i "s|^export INFERNUS_DIR=.*|export INFERNUS_DIR=\"${THISDIR}\"|" ~/.bashrc
	echo "Updated INFERNUS_DIR in ~/.bashrc"
fi

#add user input to confirm that they want to install into the current environment
# read -p "Do you want to install Infernus into the current environment? (y/n): " confirm
# if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
# 	echo "Installation halted. Please ensure you have activated a Python virtual environment for installing Infernus."
# 	exit 1
# fi

# pip install .

#check if GWSamplegen is already installed
#isInstalled=$(pip show GWSamplegen | grep -c "Name: GWSamplegen")
isInstalled=0
if [[ $isInstalled -eq 0 ]]; then
	echo "GWSamplegen is not installed. It will be installed into the parent directory of Infernus."
	read -p "Do you want to install GWSamplegen into $(dirname $INFERNUS_DIR)/GWSamplegen? (y/n): " confirm
	if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
		echo "Skipping GWSamplegen installation. Note that Infernus will not work without GWSamplegen."
		exit 0
	fi
	#install GWSamplegen from GitHub
	echo "Installing GWSamplegen from GitHub..."
	#put it in the parent directory of the current working directory
	git clone git@github.com:alistair-mcleod/GWSamplegen.git ../GWSamplegen
	cd ../GWSamplegen
	#pip install .
	cd "${INFERNUS_DIR}"
fi

#also add the virtual environment as INFERNUS_ENV
if ! grep -q "export INFERNUS_ENV=" ~/.bashrc; then
	echo "Appending INFERNUS_ENV to ~/.bashrc"
	echo "export INFERNUS_ENV=\"${VIRTUAL_ENV}/bin/activate\"" >> ~/.bashrc
else
	echo "INFERNUS_ENV already set in ~/.bashrc. Overwriting..."
	#overwrite the existing line to ensure it points to the correct virtual environment
	sed -i "s|^export INFERNUS_ENV=.*|export INFERNUS_ENV=\"${VIRTUAL_ENV}/bin/activate\"|" ~/.bashrc
	echo "Updated INFERNUS_ENV in ~/.bashrc"
fi

echo "Installation of Infernus complete. Please restart your terminal or run 'source ~/.bashrc' to update your environment."
