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
read -p "Do you want to install Infernus into the current environment? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
	echo "Not installing into a virtual environment. Please ensure you have a virtual environment with Infernus installed."
	#echo "Installation halted. Please ensure you have activated a Python virtual environment for installing Infernus."
	#exit 1
else
	pip install .
fi


#check if GWSamplegen is already installed
#note there's two things to check: if the venv already has GWSamplegen, and if the parent directory has GWSamplegen
isInVenv=$(pip show GWSamplegen | grep -cw "Name: GWSamplegen")
isInParentDir=$(ls $(dirname $INFERNUS_DIR) | grep -cw "GWSamplegen")
# isInVenv=0
# isInParentDir=0
echo "Checking for GWSamplegen installation..."

if [[ $isInParentDir -eq 0 ]]; then
	read -p "Do you want to clone GWSamplegen into $(dirname $INFERNUS_DIR)/GWSamplegen? (y/n): " confirm
	if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
		echo "Skipping GWSamplegen installation in parent directory."
		if [[ $isInVenv -eq 0 ]]; then
			echo "GWSamplegen is not installed in the current virtual environment. Note Infernus will not work without GWSamplegen."
		fi
		#exit 0
	else
		echo "Installing GWSamplegen from GitHub..."
		#put it in the parent directory of the current working directory
		git clone git@github.com:alistair-mcleod/GWSamplegen.git ../GWSamplegen
		#add the GWSamplegen directory as GWSAMPLEGEN_DIR in ~/.bashrc
		if ! grep -q "export GWSAMPLEGEN_DIR=" ~/.bashrc; then
			echo "Appending GWSAMPLEGEN_DIR to ~/.bashrc"
			echo "export GWSAMPLEGEN_DIR=\"$(dirname $INFERNUS_DIR)/GWSamplegen\"" >> ~/.bashrc
		else
			echo "GWSAMPLEGEN_DIR already set in ~/.bashrc. Overwriting..."
			#overwrite the existing line to ensure it points to the correct directory
			sed -i "s|^export GWSAMPLEGEN_DIR=.*|export GWSAMPLEGEN_DIR=\"$(dirname $INFERNUS_DIR)/GWSamplegen\"|" ~/.bashrc
			echo "Updated GWSAMPLEGEN_DIR in ~/.bashrc"
		fi
	fi

	if [[ $isInVenv -eq 0 ]]; then
	read -p "Do you want to install GWSamplegen into the current virtual environment? (y/n): " confirm
		if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
			echo "Skipping GWSamplegen installation in "
		else
			echo "Installing..."
			cd ../GWSamplegen
			bash install.sh
			pip install .
			cd "${INFERNUS_DIR}"
		fi
	fi
fi

if [[ $isInVenv -eq 1 && $isInParentDir -eq 1 ]]; then
	echo "GWSamplegen is already installed in both the current virtual environment and the parent directory. Skipping installation."
fi


#also add the virtual environment as INFERNUS_ENV. 
#TODO: This is deprecated and should be removed
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
