# CSAW-HACK-ML
ECE GY 9163 Final Project

Anup Upasani (asu224)

Neeraja Narayanswamy (nn2108)

Priyanka Shishodia (ps4118)


## Instructions

## Step 1: Downloading Data

Go into the data folder herein and copy the link in the Google Drive Link text file or get the link from below.

Download all data as is and put it into the data folder

Link to data files : https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing

## Step 2: Checking Imports

The following imports are required.

	keras, tensorflow, h5py, numpy, matplotlib, random, opencv-python, datetime, scipy, imageio
	
Note that sys, shutil, and math are required as well, but these should already be part of the python installation

## Step 2.5 (Optional): Checking Code

To see how the repair.py code generates test images, run the following command using a command prompt inside the eval folder herein
		
	cd eval/
	python3 repair.py
		
Note that this will take only a few minutes to run at maximum

To see how the repair.py code generates GoodNet models, run the following command using a command prompt inside the eval folder herein
		
	python3 repair.py init
		
Note that this will take 30+ minutes to run
		
To see how the repair.py code generates GoodNet models and sets up STRIP, run the following command using a command prompt inside the eval folder herein
	
	python3 repair.py init complex
		
Note that this will take 4+ hours to run
		
## Step 3: Evaluating Individual Images

Note: Please place any NEW images to test inside the eval folder herein (NOT inside eval/poisoned_images). Note that eval/poisoned_images contains pre-generated poisoned images to test the eval scripts on
	
To evaluate an image with any eval script, use the following syntax, where items in brackets are user inputs specified below. Run the command using a command prompt inside the eval folder herein
	
1. Pre-generated poisoned images:
	
		python3 [eval_script] poisoned_images/[image]

		Where,
			[eval_script] can be the following options:

				eval_sunglasses.py, eval_anon1.py, eval_anon2.py, eval_mtmt.py

			(Which corresponds to the sunglasses, anonymous 1, anonymous 2, and multi-target multi-trigger networks, respectively)

			[image] can be any of the filenames (with extension) inside the eval/poisoned images
			

For example,

	python3 eval_sunglasses.py poisoned_images/poisonres_sunglasses.png
		
2. New images (clean or poisoned):
		
		python3 [eval_script] [image]
			
		Where, 
			[eval_script] can be the following options

				eval_sunglasses.py, eval_anon1.py, eval_anon2.py, eval_mtmt.py

			(Which correspond to the sunglasses, anonymous 1, anonymous 2, and multi-target multi-trigger networks, respectively)
				
			[image] can be any of the filenames (with extension) that were added to the eval folder
		
		

