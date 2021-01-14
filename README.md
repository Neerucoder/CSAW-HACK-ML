# CSAW-HACK-ML
ECE GY 9163 Final Project

Anup Upasani (asu224)

Neeraja Narayanswamy (nn2108)

Priyanka Shishodia (ps4118)


Anup's notes:

install all imports that break
[Optional] run repair.py with init mode on and optional complex mode
	this will create goodnet models to use later - they already exist
run each eval script with the correct syntax
change sunglasses_bd_net.h5 for multi-target multi-trigger set to mtmtsunglasses_bd_net.h5

todo: 	create 3 more eval scripts -- please check the 3 eval files uploaded --> eval_anonymous1.py,eval_anonymous2.py,eval_mtmt.py and make the needed changes
	put link to data files
Link to data files : https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing



Instructions to run the code:

1. Download and open the ML_Final_Project.ipynb on google colab
2. Run the "Imports" and "Functions" section
3. Run the first two cells of the "Testing the Network" section(These cells clone the original repository and mount the your personal google drive onto the notebook)
4. From the following drive, upload the contents of data/ folder into your own google drive: https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab
5. In the third cell of the "Testing the Network" with the comment 'file transfer' change the '/content/drive/MyDrive/MLSecurityProjectData/main_data/' address to the address where you have stored the files from step 4 in your own drive, for example : '/content/drive/MyDrive/MLSecurityProjectData/main_data/clean_test_data.h5' should be changed to 'Your Address/clean_test_data.h5'
6. Run the remaining cells in order to test our algorithm
