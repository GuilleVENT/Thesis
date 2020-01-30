## general
import os
from os import listdir, path
import pandas as pd
import numpy as np
import numpy.matlib as matlib
from itertools import chain
import json

## coloring
from termcolor import colored

## sklearn
from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

## used for parameter selection
from sklearn import svm,  model_selection # grid search
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

## Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

## ensemble
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

## metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

## Principal Component Analyze
from sklearn.decomposition import PCA

## to save parameters and classifiers
import pickle# as pickle

## Plotting
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab
## for PCA+SVD distance
from scipy.spatial import distance

## debugging
import sys

## global directories:
pl_data = '/Users/guillermoventuramartinezAIR/Desktop/FP/PL_DATA/'

path_RES = r'user_PL/'
def PL_selector():	## selects the playlist out of the info file. 
	
	
	A = ('spotify','RapCaviar','37i9dQZF1DX0XUsuxWHRQd')
	
	B = ('spotify','mint','37i9dQZF1DX4dyzvuaRJ0n')
	
	C = ('spotify',"Today's Top Hits",'37i9dQZF1DXcBWIGoYBM5M')
	
	D = ('spotify','Rock Classics','37i9dQZF1DWXRqgorJj26U')
	
	E = ('spotify_uk_','Massive Dance Hits','37i9dQZF1DX1N5uK98ms5p')
	
	F = ('spotify','¡Viva Latino!','37i9dQZF1DX10zKzsJ2jva')	

	G = ('spotify','Peaceful Piano','37i9dQZF1DX4sWSpwq3LiO')
	
	H = ('spotify_france','Pop Urbaine','37i9dQZF1DWYVURwQHUqnN')
	
	J = ('spotify_germany','Techno Bunker','37i9dQZF1DX6J5NfMJS675')
	
	K = ('topsify','100 Uplifting Songs','6Qf2sXTjlH3HH30Ijo6AUp')

	I = ('spotify','Hot Country','37i9dQZF1DX1lVhptIYRda')

	AA = ('topsify', 'Pop Royalty', '5iwkYfnHAGMEFLiHFFGnP4')
	
	AB = ('topsify', 'Hip-Hop R&B Nation', '7kdOsNnHtzwncTBnI3J17w')
	
	AC = ('spotify_uk_', 'Yoga & Meditation', '37i9dQZF1DX9uKNf5jGX6m')
	
	AD =  ('spotify_uk_', 'Teen Party', '37i9dQZF1DX1N5uK98ms5p')

	AF = ('spotify_uk_', 'Tropical House', '37i9dQZF1DX0AMssoUKCz7')

	AG = ('spotify_uk_', 'Feel Good Friday', '37i9dQZF1DX1g0iEXLFycr')
	
	AG = ('topsify', 'House Music 2020', '2otQLmbi8QWHjDfq3eL0DC')
	
	BA = ('topsify', '100 Uplifting Songs', '6Qf2sXTjlH3HH30Ijo6AUp')
	
	BB = ('topsify', 'UK Top 50', '1QM1qz09ZzsAPiXphF1l4S')
	
	BC = ('topsify', 'House Music 2020', '2otQLmbi8QWHjDfq3eL0DC') 
	
	BD = ('topsify', '100 Uplifting Songs', '6Qf2sXTjlH3HH30Ijo6AUp')
	
	BE = ('topsify', 'UK Top 50', '1QM1qz09ZzsAPiXphF1l4S')

	CA = ('spotify','All Out 80s','37i9dQZF1DX4UtSsGT1Sbe')

	CB = ('spotify','All Out 00s','37i9dQZF1DX4o1oenSJRJd')

	CC = ('chillhopmusic','lofi hip hop beats - music to study/relax to ( lo-fi chill hop )','74sUjcvpGfdOvCHvgzNEDO')

	CD = ('spotify_france','Hits du Moment','37i9dQZF1DWVuV87wUBNwc')

	if True:
		pl_list1_1 = [A,J,G,F,I] # B,F] 
		pl_list1_2 = [A,G,F,I]
		pl_list1_3 = [A,J,F,I]
		pl_list1_4 = [J,G,F,I]

		pl_list2_1 = [CA,CB,CC,AC,E]
		pl_list2_2 = [CA,CC,AC,E]
		pl_list2_3 = [CB,CC,E]
		
		pl_list3_1 = [CA,C,J,F]

		pl_list_4_1 = [CC,CA,AG,A]#,G,F]
		pl_list_4_2 = [CC,CA,AG,A,G,F]
		pl_list_4_3 = [CC,AG,AF,F,A]
		pl_list_4_4 = [CC,AG,B,F]
		
		pl_list_5_1 = [H,CD,J,G,F]
		pl_list_5_2 = [CD,J,G,F]
		
		#pl_list4 = [AF,AD,G,F]
		#pl_list5 = [AA, AF, AC, G, A]
		#pl_list6 = [G,AC,I,AF]
		#pl_list = [('topsify', 'Pop Royalty', '5iwkYfnHAGMEFLiHFFGnP4'), ('topsify', 'Hip-Hop R&B Nation', '7kdOsNnHtzwncTBnI3J17w'), ('topsify', 'House Music 2020', '2otQLmbi8QWHjDfq3eL0DC'), ('topsify', '100 Uplifting Songs', '6Qf2sXTjlH3HH30Ijo6AUp'), ('topsify', 'UK Top 50', '1QM1qz09ZzsAPiXphF1l4S')]
		#pl_list = [('spotify_uk_', 'Yoga & Meditation', '37i9dQZF1DX9uKNf5jGX6m'), ('spotify_uk_', 'Massive Dance Hits', '37i9dQZF1DX5uokaTN4FTR'), ('spotify_uk_', 'Teen Party', '37i9dQZF1DX1N5uK98ms5p')]#, ('spotify_uk_', 'Tropical House', '37i9dQZF1DX0AMssoUKCz7'), ('spotify_uk_', 'Feel Good Friday', '37i9dQZF1DX1g0iEXLFycr')]

	pl_sets = [pl_list1_1,pl_list1_2,pl_list1_3,pl_list1_4,pl_list2_1,pl_list2_2,pl_list2_3,pl_list3_1,pl_list_4_1,pl_list_4_2,pl_list_4_3,pl_list_4_4,pl_list_5_1,pl_list_5_2]
		#, pl_list2, pl_list3, pl_list4, pl_list5, pl_list6]

	for pl_indx, classification_set in enumerate(pl_sets):
		get_X_and_y(pl_indx, classification_set)

#									MODEL = SVM/RF/kNN
def save_npy(X,X_val,y,y_val,DataSet,MODEL,pl_indx,classification_set):
	print('- saving Training/Validation Matrixes:')
	print(pl_indx)
	print(classification_set)
	print(DataSet)
	print(MODEL)
	path__ = '/Users/guillermoventuramartinezAIR/Desktop/FP/SETS/'+str(pl_indx)+'/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__ = path__+MODEL+'/'
	if not os.path.exists(path__):
		os.makedirs(path__)

	path__DS = path__+DataSet+'/'
	if not os.path.exists(path__DS):
		os.makedirs(path__DS)
	
	path__T = path__DS+'Training/'
	path__V = path__DS+'Validation/'
	
	if not os.path.exists(path__T):
		os.makedirs(path__T)
	
	if not os.path.exists(path__V):
		os.makedirs(path__V)


	## THINGS TO SAVE: X,X_val,y,y_val
	## save Training 
	file = path__T+'X_train'
	np.save(file, X, allow_pickle=True)

	file = path__T+'y_train'
	np.save(file, y, allow_pickle=True)

	file = path__V+'X_val'
	np.save(file, X_val, allow_pickle=True)

	file = path__V+'y_val'
	np.save(file, y_val, allow_pickle=True)

## takes a list of tuples as upthere as input
def get_X_and_y(pl_indx, pl_list):

	## init X and y 
	y_MIR = [] 
	y_SPO = []
	y_LYR = []

	y_ALL = []

	X_MIR = []
	X_SPO = []
	X_LYR = []

	X_ALL = []

	for index, pl in enumerate(pl_list):

		## unpack tuples
		user = pl[0]
		pl_name = pl[1]
		pl_id = pl[2]


		## open all feature sets:
		f_mir = pl_data+user+'/'+pl_id+'/MIRaudio_features.tsv'
		f_sp  = pl_data+user+'/'+pl_id+'/Spotify_features.tsv'
		f_ly  = pl_data+user+'/'+pl_id+'/Lyrics_features.tsv'
			
		MIR_df = pd.read_csv(f_mir,sep='\t').set_index('Song_ID')
		Sp_df  = pd.read_csv(f_sp,sep='\t').set_index('Song_ID')
		Lyr_df = pd.read_csv(f_ly,sep='\t').set_index('Song_ID')

		## combine feature sets into one
		ALL_df = pd.concat([MIR_df,Sp_df,Lyr_df],axis=1,sort=False)
		
		## dropping rows with NaN
		MIR_df = MIR_df.dropna()
		Sp_df  = Sp_df.dropna()
		Lyr_df = Lyr_df.dropna()

		ALL_df = ALL_df.dropna()

		#print(MIR_df)
		#print(Sp_df)
		#print(Lyr_df)
		#print(ALL_df)

		## DF to NP
		MIR_np = MIR_df.to_numpy()
		Sp_np  = Sp_df.to_numpy()
		Lyr_np = Lyr_df.to_numpy()
		ALL_np = ALL_df.to_numpy()


		## creating X-matrix
		if len(X_MIR)==0: ## init
			X_MIR = MIR_np
			X_SPO = Sp_np
			X_LYR = Lyr_np
			X_ALL = ALL_np

		else:
			X_MIR = np.vstack((X_MIR,MIR_np))
			X_SPO = np.vstack((X_SPO,Sp_np))
			X_LYR = np.vstack((X_LYR,Lyr_np))
			X_ALL = np.vstack((X_ALL,ALL_np))

		## creating y-vector
		## for each feature sets

		## Lyrics-features
		size = Lyr_np.shape
		y_ = [index] * size[0]
		y_LYR.append(y_)

		## Spotify-features
		size = Sp_np.shape
		y_ = [index] * size[0]
		y_SPO.append(y_)

		## MIR-features
		size = MIR_np.shape
		y_ = [index] * size[0]
		y_MIR.append(y_)

		size = ALL_np.shape
		y_ = [index]*size[0]
		y_ALL.append(y_)

	y_MIR = list(chain(*y_MIR))
	y_SPO = list(chain(*y_SPO))
	y_LYR = list(chain(*y_LYR))
	y_ALL = list(chain(*y_ALL))

	y_MIR = np.array([y_MIR]).T
	y_SPO = np.array([y_SPO]).T
	y_LYR = np.array([y_LYR]).T
	y_ALL = np.array([y_ALL]).T

	'''
	## Standard Scaler:
	SS = StandardScaler()
	SS_MIR = SS.fit(X=X_MIR)
	SS_SPO = SS.fit(X=X_SPO)
	SS_LYR = SS.fit(X=X_LYR)
	SS_ALL = SS.fit(X=X_ALL)
	
	X_MIR_ =  SS_MIR.transform()
	X_SPO_ =  SS_SPO.transform(X_SPO)
	X_LYR_ =  SS_LYR.transform(X_LYR)
	X_ALL_ =  SS_ALL.transform(X_ALL)

	X_MIR =  X_MIR_
	X_SPO =	 X_SPO_
	X_LYR =  X_LYR_
	X_ALL =  X_ALL_
	'''
	print("- Shapes MIR: ")
	print(y_MIR.shape)
	print(X_MIR.shape)

	print("- Shapes Spotify: ")
	print(y_SPO.shape)
	print(X_SPO.shape)

	print("- Shapes Lyrics: ")
	print(y_LYR.shape)
	print(X_LYR.shape)

	print("- Shapes Combined Feature Set")
	print(y_ALL.shape)
	print(X_ALL.shape)

	## SPLIT FOR VALIDATION:
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.15)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.15)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.15)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.15)
	
	save_npy(X_MIR, X_MIR_val, y_MIR, y_MIR_val,'MIR','kNN',pl_indx,pl_list)
	save_npy(X_SPO, X_SPO_val, y_SPO, y_SPO_val,'SPO','kNN',pl_indx,pl_list)
	save_npy(X_LYR, X_LYR_val, y_LYR, y_LYR_val,'LYR','kNN',pl_indx,pl_list)
	save_npy(X_ALL, X_ALL_val, y_ALL, y_ALL_val,'ALL','kNN',pl_indx,pl_list)

	## PCA + SVD 
	#X_MIR = do_pca(X_MIR)
	#X_ALL = do_pca(X_ALL)
	
	## train different models with 
	## this paer trains the parameteres for all the different models FR SVM kNN


	## Select KFOLD! CV 
	## if user input use these 2 lines:
	'''Kfold_text = input ("Enter the # of K-Fold Crossvalidation: ")
	num_folds = int(Kfold_text)'''
	## else:      
	num_folds = 5  ##  !! KFOLD SELECT 

	param_folder = '/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_SET_'+str(pl_indx)+'/'
	
	if not os.path.exists(param_folder):
		os.makedirs(param_folder)
	if not os.path.exists(param_folder+'MIR/'):
		os.makedirs(param_folder+'MIR/')
	if not os.path.exists(param_folder+'SPO/'):
		os.makedirs(param_folder+'SPO/')
	if not os.path.exists(param_folder+'LYR/'):
		os.makedirs(param_folder+'LYR/')
	if not os.path.exists(param_folder+'ALL/'):
		os.makedirs(param_folder+'ALL/')

	#LYR,SPO,MIR,ALL
	## kNN - MIR 
	param_file = param_folder+'/MIR/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	MIR:')
		model_parameters_kNN_MIR = open_parameters('kNN','MIR',pl_indx)#,y_MIR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	MIR:')
		model_parameters_kNN_MIR = perform_grid_search('kNN',X_MIR,y_MIR,num_folds)
		save_parameters(param_folder,model_parameters_kNN_MIR,'kNN','MIR',pl_indx)	

	## kNN - SPO
	param_file = param_folder+'/SPO/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	SPO:')
		model_parameters_kNN_SPO = open_parameters('kNN','SPO',pl_indx)#X_SPO,y_SPO,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	SPO:')
		model_parameters_kNN_SPO = perform_grid_search('kNN',X_SPO,y_SPO,num_folds)
		save_parameters(param_folder,model_parameters_kNN_SPO,'kNN','SPO',pl_indx)	

	## kNN - LYR
	param_file = param_folder+'/LYR/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	LYR:')
		model_parameters_kNN_LYR = open_parameters('kNN','LYR',pl_indx)#X_LYR,y_LYR,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	LYR:')
		model_parameters_kNN_LYR = perform_grid_search('kNN',X_LYR,y_LYR,num_folds)
		save_parameters(param_folder,model_parameters_kNN_LYR,'kNN','LYR',pl_indx)		

	## kNN - ALL 
	param_file = param_folder+'/ALL/'+'kNN.pkl'
	if path.exists(param_file):
		print(' LOADING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	ALL:')
		model_parameters_kNN_ALL = open_parameters('kNN','ALL',pl_indx)#X_ALL,y_ALL,num_folds)
	else:
		print(' COMPUTING PARAMETERS:')
		print('	kNN - Parameter Models')
		print('	ALL:')
		model_parameters_kNN_ALL = perform_grid_search('kNN',X_ALL,y_ALL,num_folds)
		save_parameters(param_folder,model_parameters_kNN_ALL,'kNN','ALL',pl_indx)		

	'''
	else:
		print(' # # # # # # ')
		print('- Loading all parameters saved:')
		model_parameters_kNN_MIR = open_parameters('kNN','MIR',pl_indx)
		model_parameters_kNN_SPO = open_parameters('kNN','SPO',pl_indx)
		model_parameters_kNN_LYR = open_parameters('kNN','LYR',pl_indx)
		model_parameters_kNN_ALL = open_parameters('kNN','ALL',pl_indx)
	'''

	
	print(' # # # # # # # # # # # # # # # # # # ')
	print('')
	print(' USING PARAMETERS FROM GRID_SEARCH ')
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	

	kNN_clf_MIR, score = train_kNN(X_MIR, y_MIR, num_folds, model_parameters_kNN_MIR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_MIR,'kNN','MIR', pl_indx)
	kNN_clf_SPO, score = train_kNN(X_SPO,y_SPO, num_folds, model_parameters_kNN_SPO)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_SPO,'kNN','SPO', pl_indx)
	kNN_clf_LYR ,score = train_kNN(X_LYR,y_LYR, num_folds, model_parameters_kNN_LYR)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_LYR,'kNN','LYR', pl_indx)
	kNN_clf_ALL, score = train_kNN(X_ALL, y_ALL, num_folds, model_parameters_kNN_ALL)
	print('- BEST SCORE:')
	print(score)
	print('')
	save_clf(kNN_clf_ALL,'kNN','ALL', pl_indx)

	print(" - VALIDATION: ")
	print('')
	print(' # # # # # # # # # # # # # # # # # # ')
	## 		SPOTIFY DATASET 
	## 		kNN
	y_true, y_pred = y_SPO_val, kNN_clf_SPO.predict(X_SPO_val)
	print_save(pl_indx,pl_list,'SPOTIFY DATASET',' kNN ',y_true, y_pred, X_SPO_val, y_SPO_val, model_parameters_kNN_SPO)

	## 		LYRICS 	DATASET 
	## 		kNN
	y_true, y_pred = y_LYR_val, kNN_clf_LYR.predict(X_LYR_val)
	print_save(pl_indx,pl_list,'LYRICS DATASET',' kNN ',y_true, y_pred, X_LYR_val, y_LYR_val, model_parameters_kNN_LYR)

	## 		MIR DATASET
	## 		kNN
	y_true, y_pred = y_MIR_val, kNN_clf_MIR.predict(X_MIR_val)
	print_save(pl_indx,pl_list,'  MIR  DATASET',' kNN ',y_true, y_pred, X_MIR_val, y_MIR_val, model_parameters_kNN_MIR)

	## 		ALL DATASET 
	##		kNN
	y_true, y_pred = y_ALL_val, kNN_clf_ALL.predict(X_ALL_val)
	print_save(pl_indx,pl_list,'  ALL  DATASET',' kNN ',y_true, y_pred, X_ALL_val, y_ALL_val, model_parameters_kNN_ALL)




def train_kNN(X,y,num_folds,params):
	clfs = []  ## for best selection

	cms = []	## multiple confussion matrixes
	
	scores = [] 	## for the results
	train_errors = []
	test_errors = []
	training_scores = []
	testing_scores = []


	## for later analysing:
	X_test_l = []
	y_test_l = []

	y = y.ravel()

	scaler = StandardScaler()
	#using 10-fold
	kf = KFold(n_splits= num_folds ,shuffle=True)

	for train_index, test_index in kf.split(X):
		#print("TRAIN:\n", train_index)
		#print("TEST:\n", test_index)
		#print(type(train_index))
		#print(type(test_index))
		print('Training kFold')
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

		clf = KNeighborsClassifier()

		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		scores.append(test_score)

		training_scores.append(train_score)
		testing_scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)

		X_test_l.append(X_test)
		y_test_l.append(y_test)

		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		clfs.append(clf)

		cms.append(cm)

	## score of best clf
	best_score, indx = max((val, idx) for (idx, val) in enumerate(scores))
	
	best_clf = clfs[indx]

	#best_clf.get_params()

	
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cms))
	print(" - - - - - - - - - - - ")
	print(" - - - - - - - - - - - ")
	print(" BEST CLASSIFICATION SCORE:")
	print("	"+str(best_score*100)+"%")
	print(" - - - - - - - - - - - ")
	print("")
	print("SCORES IN TRAINING: " )
	print("mean ", np.mean(training_scores))
	print("std  ", np.std(training_scores))
	print(training_scores)
	print(" - - - - - - - - - - - " )
	print("Min Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of kNN :")
	print("")
	print(best_clf.get_params())
	print("")
	print(" - - - - - - - - - - - ")

	return best_clf, best_score



def perform_grid_search(model, X, y, num_folds):
	y = np.ravel(y)

	# Split the dataset in two equal parts
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

	if model == 'SVM':
		## key :
		model = svm.SVC()

		# Set the parameters for cross-validation
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000],'C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]},
							{'kernel': ['sigmoid'],'gamma': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000], 'C': [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]}]
		
	
		grid_search_ = GridSearchCV(model, param_grid=tuned_parameters, iid=False, cv=num_folds)
		grid_search_.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(grid_search_.best_params_)
		print("{:.2%}".format(grid_search_.best_score_))
		print()
		
		##'''
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		print("Detailed classification report:")
		print("	- Grid scores on development set:")
		print()
		means = grid_search_.cv_results_['mean_test_score']
		stds = grid_search_.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
			print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
			print(' * * * * * * * * * * * * * * *')
		print('')
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		##'''
	
		best_clf = grid_search_.best_estimator_
		y_true, y_pred = y_test, best_clf.predict(X_test)

		print(' --> Validation: ')
		print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
		print(classification_report(y_true, y_pred))
		print(confusion_matrix(y_true,y_pred))
		print()

		res_parameters = best_clf.get_params
		
		print(res_parameters)

		return res_parameters

	if model == "RF":
		model = RandomForestClassifier()

		## TO DO
		## parameters to try out: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
		##
		'''param_grid = {'bootstrap': [True],
					'max_depth': [1,10,50,80, 90, 100],
					'max_features': [2, 3],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					'n_estimators': [1,10,50,100, 200, 300, 500, 1000],
					}'''
		## for now:
		param_grid = {'bootstrap': [True],
					'max_depth': [1,10,20,50,100,100],#[1,10,100],
					'max_features': [2],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					#'n_estimators': [1,10] ## leaving this one out for now 
					}
		##
		##


		grid_search_ = GridSearchCV(estimator = model, param_grid = param_grid, cv = num_folds, n_jobs = -1)

		grid_search_.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(grid_search_.best_params_)
		print("{:.2%}".format(grid_search_.best_score_))
		print()
		
		##'''
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		print("Detailed classification report:")
		print("	- Grid scores on development set:")
		print()
		means = grid_search_.cv_results_['mean_test_score']
		stds = grid_search_.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
			print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
			print(' * * * * * * * * * * * * * * *')
		print('')
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		##'''
		
		print(' --> Validation: ')

		best_clf = grid_search_.best_estimator_
		y_true, y_pred = y_test, best_clf.predict(X_test)

		print(' VALIDATION SET:')
		print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
		print(classification_report(y_true, y_pred))
		print(confusion_matrix(y_true,y_pred))
		print()

		res_parameters = best_clf.get_params
		print(res_parameters)
		
		return res_parameters

	if model == 'kNN':
		model = KNeighborsClassifier()

		## TO DO
		## parameters to try out: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
		##
		param_grid_top = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
             	}

		param_grid = {'n_neighbors': [3,4,5,6,7,5,10,15,20],
					'weights': ["uniform","distance"],
					'algorithm': ['auto'],#'ball_tree','kd_tree','brute'],
					'leaf_size': [1,5,10,20,30,40,50,100],#,30],
					}
		##
		##

		grid_search_ = GridSearchCV(estimator = model, param_grid = param_grid, cv = num_folds, n_jobs = -1)

		#print(type(grid_search_))

		gs_results= grid_search_.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(grid_search_.best_params_)
		print("{:.2%}".format(grid_search_.best_score_))
		print()
		
		##'''
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		print("Detailed classification report:")
		print("	- Grid scores on development set:")
		print()
		means = grid_search_.cv_results_['mean_test_score']
		stds = grid_search_.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, grid_search_.cv_results_['params']):
			print("- mean: %0.3f \n - std: (+/-%0.03f) \n - parameters: %r"% (mean, std * 2, params))
			print(' * * * * * * * * * * * * * * *')
		print('')
		print(' -- -- -- -- -- -- -- -- -- -- -- ')
		print('')
		##'''
		
		print(' --> Validation: ')

		best_clf = grid_search_.best_estimator_
		y_true, y_pred = y_test, best_clf.predict(X_test)

		print(' VALIDATION SET:')
		print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
		print(classification_report(y_true, y_pred))
		print(confusion_matrix(y_true,y_pred))
		print()

		res_parameters = best_clf.get_params
		print(res_parameters)
		
		return res_parameters


def save_parameters(param_folder,params,clf,dataset,pl_indx):
	
	#parameters 	= params()
	#print(parameters)
	#print(type(parameters))

	#	'/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_Set_'+str(pl_indx)+'/'
	param_folder = '/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_SET_'+str(pl_indx)+'/'
	param_folder_DS = param_folder+dataset+'/'

	if not path.exists(param_folder_DS):
		os.mkdir(param_folder_DS)

	param_file = param_folder_DS+clf+'.pkl'

	with open(param_file,"wb") as file:
		pickle.dump(params, file, pickle.HIGHEST_PROTOCOL)
	

def open_parameters(clf, dataset,pl_indx):
	#	'/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/PL_Set_'+str(pl_indx)+'/'

	param_file = '/Users/guillermoventuramartinezAIR/Desktop/FP/parameters/'+'PL_SET_'+str(pl_indx)+'/'+dataset+'/'+clf+'.pkl'

	file = open(param_file, 'rb')
	parameters = pickle.load(file)
	file.close()
	
	return parameters


def save_clf(clf,clf_type,dataset,pl_indx):

	if not path.exists('/Users/guillermoventuramartinezAIR/Desktop/FP/classifiers/'+'PL_SET_'+str(pl_indx)+'/'):
		os.mkdir('/Users/guillermoventuramartinezAIR/Desktop/FP/classifiers/'+'PL_SET_'+str(pl_indx)+'/')	

	clf_dir = '/Users/guillermoventuramartinezAIR/Desktop/FP/classifiers/'+'PL_SET_'+str(pl_indx)+'/'+dataset+'/'

	if not path.exists(clf_dir):
		os.mkdir(clf_dir)

	clf_file = clf_dir+clf_type+'.pkl'

	with open(clf_file,"wb") as file:
		pickle.dump(clf, file, pickle.HIGHEST_PROTOCOL)

def do_pca(X):
	print("*********** SVD + PCA ***********")

	print("SVD")
	value = np.linalg.svd(X,compute_uv=False)
	print(value)
	print("SORTED")
	value_s = sorted(value)
	print(value_s)
	print("INVERT")
	values_si = value_s[::-1]
	print(values_si)

	sum_ = []
	

	i=0
	while i < len(values_si): 
		#print i
		if i == 0:
			sum_.append(values_si[i])
		
		else:
			sum_.append(sum_[i-1]+values_si[i])

		i += 1

	print("SUM")
	print(sum_)

	## gerade
	#gerade = range(int(sum_[0]),int(sum_[-1]),int(sum_[-1]-sum_[0])/len(sum_))
	
	## distance
	curve = sum_
	nPoints = len(curve)
	allCoord = np.vstack((range(nPoints), curve)).T
	np.array([range(nPoints), curve])
	firstPoint = allCoord[0]
	lineVec = allCoord[-1] - allCoord[0]
	lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
	vecFromFirst = allCoord - firstPoint
	scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
	vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
	vecToLine = vecFromFirst - vecFromFirstParallel
	distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
	#global idxOfBestPoint
	idxOfBestPoint = np.argmax(distToLine)
	print(idxOfBestPoint)

	markers_on = [idxOfBestPoint]

	plt.figure(1)
	plt.plot(range(0,len(sum_)),sum_,color='g',marker='o',markevery=markers_on)
	#plt.plot(range(0,len(gerade)),gerade,color='b')
	plt.xlabel('Feature')
	plt.ylabel('Sum Singular Values')
	plt.title('SVD')
	plt.legend(['Index of bend: %d' % idxOfBestPoint],loc='center right',)
	plt.grid()
	plt.savefig('SVD-pl-sum.pdf')


	pca = PCA(n_components=idxOfBestPoint, svd_solver= 'full') 	# Doc:n_components == 'mle' and svd_solver == 'full' Minka's MLE is used to guess the dimension 
	print(X.shape)
	pca.fit(X)
	X_new = pca.transform(X)

	print(X_new)
	print(X_new.shape)
	
	
	return X_new


def print_save(pl_indx,pl_list,data_set, clf, y_true, y_pred, X_val, y_val, params):

	print(" # * # * # * # * # * # * # * #")
	print(" - PLAYLIST TRAINED TO CLF")
	print(' PL CLF index:')
	print(" - "+str(pl_indx))
	print(" - - - - - - - - - - - - - - -")
	## unpack tuples
	print(" - TRAINED WITH:")
	for i,pl in enumerate(pl_list):
		print(' - PL ##'+str(i))
		print(pl)
	print(" -------- RESULTS --------")
	print(" ----- "+data_set+"-----")
	print(' -'+clf+' Classifier ')
	print(' - PARAMETERS:')
	print(params())
	print(' - size of evaluation set')
	print(X_val.shape)
	print(y_val.shape)
	print(' --> Validation: ')
	print(' - Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()
	print(" - - - - - - - - - - - ")
	print(" ")
	print(" # * # * # * # * # * # * # * #")

	text_file = open(r"CLF_results/"+data_set+" - "+clf+".txt", "a+")
	text_file.write(" # * # * # * # * # * # * # * #\n")
	text_file.write(" - PLAYLIST TRAINED TO CLF\n")
	text_file.write(' PL CLF index:')
	text_file.write(" -" +str(pl_indx))
	text_file.write(" - - - - - - - - - - - - - - -\n")
	for i, pl in enumerate(pl_list):
		text_file.write(' - ############:\n'+str(i)+"\n")
		text_file.write(' - USER:\n')
		text_file.write(str(pl[0])+"\n")
		text_file.write(' - PLAYLIST:')
		text_file.write(str(pl[1])+"\n")
		text_file.write(' - PL ID')
		text_file.write(str(pl[2])+"\n")
	text_file.write(" -------- RESULTS --------\n")
	text_file.write(" ----- "+data_set+"-----\n")
	text_file.write(' -'+clf+' Classifier \n')
	text_file.write(' - PARAMETERS:\n')
	param_str_= json.dumps(params())
	text_file.write(param_str_)
	#text_file.write(' - size of evaluation set\n')

	#X_val_shape = X_val.shape
	#y_val_shape = y_val.shape
	
	#x_shape = ''.join(X_val_shape)
	#y_shape = ''.join(y_val_shape)
	#text_file.write(x_shape)
	#text_file.write('\n')
	#text_file.write(x_shape)
	text_file.write('\n')
	text_file.write(' --> Validation: ')
	text_file.write('\n')
	text_file.write(' - Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	text_file.write('\n')
	text_file.write(classification_report(y_true, y_pred))
	text_file.write('\n')
	text_file.write(np.array2string(confusion_matrix(y_true,y_pred)))
	text_file.write('\n')
	text_file.write(" - - - - - - - - - - - ")
	text_file.write('\n')
	text_file.write(" ")
	text_file.write('\n')
	text_file.write(" # * # * # * # * # * # * # * #")
	text_file.write('\n')
	text_file.close()

PL_selector()

