# CSAW-HACK-ML
ECE GY 9163 Final Project

Anup Upasani (asu224)

Neeraja Narayanswamy (nn2108)

Priyanka Shishodia (ps4118)


## Instructions

Step 1: Downloading Data

	Go into the data folder herein and copy the link in the Google Drive Link text file
	
	Then download all data as is and put it into the data folder

Step 1: Checking Imports

	The following imports are required.
	
	keras, tensorflow, h5py, numpy, matplotlib, random, opencv-python, datetime, scipy, imageio
	
	Note that sys, shutil, and math are required as well, but these should already be part of the python installation

Step 1.5 (Optional): Checking Code

	To see how the repair.py code generates GoodNet models, run the following command using a command prompt inside the eval folder herein
		
		python repair.py init
		
		Note that this will take 30+ minutes to run
		
	To see how the repair.py code generates GoodNet models and sets up STRIP, run the following command using a command prompt inside the eval folder herein
	
		python repair.py init complex
		
		Note that this will take 4+ hours to run

Step 2: Run Setup

	Run the following command to test code and generate test images. Use a command prompt inside the eval folder herein
	
		python repair.py
		
		Note that this should only take at maximum a few minutes
		
Step 3: Evaluating Individual Images
	
	To evaluate an image with any eval script, use the following syntax, where items in brackets are user inputs specified below. 
	
		python [eval_script] [image png]
		
		Where [eval_script] can be the following options
			
			eval_sunglasses.py, eval_anon1.py, eval_anon2.py, eval_mtmt.py


run each eval script with the correct syntax
change sunglasses_bd_net.h5 for multi-target multi-trigger set to mtmtsunglasses_bd_net.h5

todo: 	create 3 more eval scripts -- please check the 3 eval files uploaded --> eval_anonymous1.py,eval_anonymous2.py,eval_mtmt.py and make the needed changes
	put link to data files
Link to data files : https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing
