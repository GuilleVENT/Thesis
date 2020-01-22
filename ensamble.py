## general
import os
from os import listdir
import pandas as pd
import numpy as np
import numpy.matlib as matlib
from itertools import chain

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

	if True:
		#pl_list = [A,D,E,G,J] # B,F]
		#pl_list = [('topsify', 'Pop Royalty', '5iwkYfnHAGMEFLiHFFGnP4'), ('topsify', 'Hip-Hop R&B Nation', '7kdOsNnHtzwncTBnI3J17w'), ('topsify', 'House Music 2020', '2otQLmbi8QWHjDfq3eL0DC'), ('topsify', '100 Uplifting Songs', '6Qf2sXTjlH3HH30Ijo6AUp'), ('topsify', 'UK Top 50', '1QM1qz09ZzsAPiXphF1l4S')]
		pl_list = [('spotify_uk_', 'Yoga & Meditation', '37i9dQZF1DX9uKNf5jGX6m'), ('spotify_uk_', 'Massive Dance Hits', '37i9dQZF1DX5uokaTN4FTR'), ('spotify_uk_', 'Teen Party', '37i9dQZF1DX1N5uK98ms5p')]#, ('spotify_uk_', 'Tropical House', '37i9dQZF1DX0AMssoUKCz7'), ('spotify_uk_', 'Feel Good Friday', '37i9dQZF1DX1g0iEXLFycr')]


	get_X_and_y(pl_list)


def get_X_and_y(pl_list):

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
	
	X_MIR, X_MIR_val, y_MIR, y_MIR_val = train_test_split(X_MIR, y_MIR, test_size=0.05)
	X_SPO, X_SPO_val, y_SPO, y_SPO_val = train_test_split(X_SPO, y_SPO, test_size=0.05)
	X_LYR, X_LYR_val, y_LYR, y_LYR_val = train_test_split(X_LYR, y_LYR, test_size=0.05)
	X_ALL, X_ALL_val, y_ALL, y_ALL_val = train_test_split(X_ALL, y_ALL, test_size=0.05)
	

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

	
	print('	SVM - Parameter Models')
	print('	MIR:')
	model_parameters_SVM_MIR = perform_grid_search('SVM',X_MIR,y_MIR,num_folds)
	print('	SVM - Parameter Models')
	print('	SPO:')
	model_parameters_SVM_SPO = perform_grid_search('SVM',X_SPO,y_SPO,num_folds)
	print('	SVM - Parameter Models')
	print('	LYR:')
	model_parameters_SVM_LYR = perform_grid_search('SVM',X_LYR,y_LYR,num_folds)
	print('	SVM - Parameter Models')
	print('	ALL:')
	#model_parameters_SVM_ALL = perform_grid_search('SVM',X_ALL,y_ALL,num_folds)
	
	#sys.exit()
	
	print('	RF - Parameter Models')
	print('	MIR:')
	#model_parameters_RF_MIR = perform_grid_search('RF',X_MIR,y_MIR,num_folds)
	print('	RF - Parameter Models')
	print('	SPO:')
	model_parameters_RF_SPO = perform_grid_search('RF',X_SPO,y_SPO,num_folds)
	print('	RF - Parameter Models')
	print('	LYR:')
	model_parameters_RF_LYR = perform_grid_search('RF',X_LYR,y_LYR,num_folds)
	print('	RF - Parameter Models')
	print('	ALL:')
	#model_parameters_RF_ALL = perform_grid_search('RF',X_ALL,y_ALL,num_folds)
	
	#sys.exit()
	
	print('	kNN - Parameter Models')
	print('	MIR:')
	#model_parameters_kNN_MIR = perform_grid_search('kNN',X_MIR,y_MIR,num_folds)
	print('	kNN - Parameter Models')
	print('	SPO:')
	model_parameters_kNN_SPO = perform_grid_search('kNN',X_SPO,y_SPO,num_folds)
	print('	kNN - Parameter Models')
	print('	LYR:')
	model_parameters_kNN_LYR = perform_grid_search('kNN',X_LYR,y_LYR,num_folds)
	print('	kNN - Parameter Models')
	print('	ALL:')
	#model_parameters_kNN_ALL = perform_grid_search('kNN',X_ALL,y_ALL,num_folds)

	print(' # # # # # # # # # # # # # # # # # # ')
	print(' # # # # # # # # # # # # # # # # # # ')

	print(' USING TRAINED MODELS')

	print(' # # # # # # # # # # # # # # # # # # ')
	print(' # # # # # # # # # # # # # # # # # # ')


	SVM_clf_MIR = train_SVM(X_MIR, y_MIR, num_folds, model_parameters_SVM_MIR)
	#RFclf_MIR = train_RF(X_MIR, y_MIR, num_folds, model_parameters_RF_MIR)
	#kNN_clf_MIR = train_kNN(X_MIR, y_MIR, num_folds, model_parameters_kNN_MIR)
	
	SVM_clf_SPO = train_SVM(X_SPO,y_SPO, num_folds, model_parameters_SVM_SPO)
	RF_clf_SPO = train_RF(X_SPO,y_SPO, num_folds, model_parameters_RF_SPO)
	kNN_clf_SPO = train_kNN(X_SPO,y_SPO, num_folds, model_parameters_kNN_SPO)
	
	SVM_clf_LYR = train_SVM(X_LYR,y_LYR, num_folds, model_parameters_SVM_LYR)
	RF_clf_LYR = train_RF(X_LYR, y_LYR, num_folds, model_parameters_RF_LYR)
	kNN_clf_LYR = train_kNN(X_LYR,y_LYR, num_folds, model_parameters_kNN_LYR)


	print(" -------- RESULTS --------")
	print(" -----SPOTIFY DATASET-----")

	print('- SVM Classifier EVALUATION:')
	print('size of evaluation set')
	print(X_SPO_val.shape)
	print(y_SPO_val.shape)
	
	y_true, y_pred = y_SPO_val, SVM_clf_SPO.predict(X_SPO_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()

	print(" - - - - - - - - - - - ")
	print(" ")

	print('- RF Classifier EVALUATION:')
	print('size of evaluation set')
	print(X_SPO_val.shape)
	print(y_SPO_val.shape)
	
	y_true, y_pred = y_SPO_val, RF_clf_SPO.predict(X_SPO_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()
	#score = RF_clf_SPO.score(X_SPO_val, y_SPO_val)
	'''
	cm = confusion_matrix(y_SPO_val,RF_clf_SPO_y_val)
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cm))
	print(" - - - - - - - - - - - ")
	print(score)
	'''
	print(" - - - - - - - - - - - ")
	print(" ")
	print(" ")
	print('- kNN Classifier EVALUATION:')
	print('size of evaluation set')
	print(X_SPO_val.shape)
	print(y_SPO_val.shape)
	
	#kNN_clf_SPO_y_val = kNN_clf_SPO.predict(X_SPO_val)

	#score = kNN_clf_SPO.score(X_SPO_val, y_SPO_val)

	y_true, y_pred = y_SPO_val, kNN_clf_SPO.predict(X_SPO_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()


	#cm = confusion_matrix(y_SPO_val,kNN_clf_SPO_y_val)
	print(" - - - - - - - - - - - ")
	#print("CONFUSSION MATRIX:")
	#print(np.asarray(cm))
	#print(" - - - - - - - - - - - ")
	#print(score)
	print(" - - - - - - - - - - - ")




	print(" -------- RESULTS --------")
	print(" -----LYRICS DATASET------")

	print('- SVM Classifier EVALUATION:')
	print('size of evaluation set')
	print(X_LYR_val.shape)
	print(y_LYR_val.shape)
	
	y_true, y_pred = y_LYR_val, SVM_clf_LYR.predict(X_LYR_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print("")

	print(" - - - - - - - - - - - ")
	print(" ")
	
	print('- RF Classifier EVALUATION:')
	print('sive of evaluation set')
	print(X_LYR_val.shape)
	print(y_LYR_val.shape)
	
	#RF_clf_LYR_y_val = RF_clf_LYR.predict(X_LYR_val)

	#score = RF_clf_LYR.score(X_SPO_val, y_SPO_val)

	y_true, y_pred = y_LYR_val, RF_clf_LYR.predict(X_LYR_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print()

	'''
	cm = confusion_matrix(y_LYR_val,RF_clf_LYR_y_val)
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cm))
	print(" - - - - - - - - - - - ")
	print(score)
	print(" - - - - - - - - - - - ")
	'''
	print(" ")
	print(" - - - - - - - - - - -")
	print('- kNN Classifier EVALUATION:')
	print('sive of evaluation set')
	print(X_LYR_val.shape)
	print(y_LYR_val.shape)
	
	y_true, y_pred = y_LYR_val, kNN_clf_LYR.predict(X_LYR_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print("")
	'''
	kNN_clf_LYR_y_val = kNN_clf_LYR.predict(X_LYR_val)

	score = kNN_clf_LYR.score(X_LYR_val, y_LYR_val)

	cm = confusion_matrix(y_LYR_val,kNN_clf_LYR_y_val)
	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cm))
	print(" - - - - - - - - - - - ")
	print(score)
	'''
	print(" - - - - - - - - - - - ")


	print(" -------- RESULTS --------")
	print(" ----- MIR  DATASET ------")

	print('- SVM Classifier EVALUATION:')
	print('size of evaluation set')
	print(X_MIR_val.shape)
	print(y_MIR_val.shape)
	
	y_true, y_pred = y_MIR_val, SVM_clf_MIR.predict(X_MIR_val)

	print(' --> Validation: ')
	print('Accuracy: '+"{:.2%}".format(accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred))
	print(confusion_matrix(y_true,y_pred))
	print("")

	print(" - - - - - - - - - - - ")
	print(" ")


	sys.exit('f***')


	#sys.exit()
	######################
	# ensemble/voting classifier 
	#ensemble_voting(X_ALL, X_ALL_val, y_ALL, y_ALL_val)
	#eclf.fit(X_train, y_train)
	#print(eclf.score(X_test,y_test))

def ensemble_voting(X_train,X_test,y_train,y_test):
	y_train = y_train.ravel()
	y_test = y_test.ravel()

	C_value, gamma_value,kernel_type = svc_param_selection(X_train, y_train, 5)
	######################
	# fit clf1 with df1
	pipe1 = Pipeline([
		('col_extract', ColumnExtractor( cols=range(0,34) )), # selecting features 0 and 1 (df1) to be used with LR (clf1)
		('clf', SVC(C=C_value,kernel=kernel_type,gamma=gamma_value))
		])
	
	pipe1.fit(X_train, y_train) # sanity check
	print(' Sanity check')
	print(pipe1.score(X_test,y_test)) # sanity check
	

	######################
	# fit clf2 with df2
	pipe2 = Pipeline([
		('col_extract', ColumnExtractor( cols=range(35,47) )), # selecting features 2 and 3 (df2) to be used with SVC (clf2)
		('clf', KNeighborsClassifier())
		])

	pipe2.fit(X_train, y_train) # sanity check
	print(' Sanity check')
	print(pipe2.score(X_test,y_test)) # sanity check
	
	######################
	# fit clf3 with df3
	pipe3 = Pipeline([
		('col_extract', ColumnExtractor( cols=range(48,95) )), # selecting features 2 and 3 (df2) to be used with SVC (clf2)
		('clf', RandomForestClassifier(n_estimators=20, random_state=0,criterion='entropy'))
		])

	pipe3.fit(X_train, y_train) # sanity check
	print(' Sanity check')
	print(pipe3.score(X_test,y_test)) # sanity check

	######################
	# ensemble/voting classifier where clf1 fitted with df1 and clf2 fitted with df2
	eclf = VotingClassifier(estimators=[('MIR-SVM', pipe1), ('SPO-kNN', pipe2), ('LYR-RF',pipe3)], voting='hard')
	eclf.fit(X_train, y_train)
	print(eclf.score(X_test,y_test))
	

def train_SVM(X,y,num_folds,params):
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

	parameters 	= params()
	
	## leaving scaling out now
	#scaler = StandardScaler()
	
	## OLD ->#C_value, gamma_value,kernel_type = svc_param_selection(X, y, num_folds)
	
	#using 5-fold
	kf = KFold(n_splits=num_folds,shuffle=True)

	for train_index, test_index in kf.split(X):
		print("TRAIN:\n", train_index)
		print("TEST:\n", test_index)
		print(type(train_index))
		print(type(test_index))

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#scaler.fit(X_train)
		#X_train = scaler.transform(X_train)
		#X_test = scaler.transform(X_test)

		clf = SVC(C=parameters['C'],kernel=parameters['kernel'],gamma=parameters['gamma'], probability=True)
		
		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		training_scores.append(train_score)
		testing_scores.append(test_score)

		scores.append(test_score)

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
	print("Min. Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of SVM :")
	print("kernel: ", parameters['kernel'] )
	print("C     : ", parameters['C'])
	print("gamma : ", parameters['gamma'])
	print(" - - - - - - - - - - - ")
	print(" This clf will be saved with pickle at this point with the input data")
	return best_clf	


def train_RF(X,y,num_folds,params):
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

	parameters 		= params()

	scaler = StandardScaler()

	kf = KFold(n_splits= num_folds ,shuffle=True)

	for train_index, test_index in kf.split(X):
		print("TRAIN:\n", train_index)
		print("TEST:\n", test_index)
		print(type(train_index))
		print(type(test_index))

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

		clf = RandomForestClassifier(n_estimators=parameters['n_estimators'],bootstrap=parameters['bootstrap'],max_depth=parameters['max_depth'],max_features=parameters['max_features'],min_samples_leaf = parameters['min_samples_leaf'],min_samples_split = parameters['min_samples_split'],random_state=0,criterion=parameters['criterion'])
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
	print(" - - - - - - - - - - - ")
	print("Min Test Errors: ",min(test_errors))
	print(" - - - - - - - - - - - ")	
	print("SCORES IN TEST: "  )
	print("mean ",np.mean(testing_scores))
	print("std  ",np.std(testing_scores))
	print(testing_scores)
	print(" - - - - - - - - - - - ")
	print("TRAINING Parameters")
	print(" # of folds : ", num_folds)
	print(" PARAMETERS of RF :")
	print(best_clf.get_params())
	print(" - - - - - - - - - - - ")
	print(" This clf will be saved with pickle at this point with the input data")
	return best_clf

## to do parameter adaptation HERE: 
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
		print("TRAIN:\n", train_index)
		print("TEST:\n", test_index)
		print(type(train_index))
		print(type(test_index))

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

	return best_clf



## TUNING VALUES C AND GAMMA IN RBF 
##		To-do:
##		https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
##		https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
##		exhaustive grid search: https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search
def svc_param_selection(X, y, num_folds):
	kernel_type = 'rbf'
	print(" ...")
	print(" CALCULATIN C and GAMMA in KERNEL "+kernel_type)

	Cs = [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]
	gammas = [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000]
	param_grid = {'C': Cs, 'gamma' : gammas}

	grid_search_ = model_selection.GridSearchCV(svm.SVC(kernel=kernel_type), param_grid, cv=num_folds,iid=True)
	grid_search_.fit(X, y)
	grid_search_.best_params_
	#return grid_search_.best_params_
	print(grid_search_.best_params_)
	print(type(grid_search_.best_params_))
	print("C ",grid_search_.best_params_['C'])
	print("GAMMA " ,grid_search_.best_params_['gamma'])
	
	return(grid_search_.best_params_['C'],grid_search_.best_params_['gamma'],kernel_type)

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
		param_grid_top = {'bootstrap': [True],
					'max_depth': [1,10,50,80, 90, 100],
					'max_features': [2, 3],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					'n_estimators': [1,10,50,100, 200, 300, 500, 1000],
					}
		## for now:
		param_grid = {'bootstrap': [True],
					'max_depth': [1,10,20],#[1,10,100],
					'max_features': [2],
					'min_samples_leaf': [3, 4, 5],
					'min_samples_split': [8, 10, 12],
					'n_estimators': [1,10],
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

		param_grid = {'n_neighbors': [3,5,10,15,20],
					'weights': ["uniform","distance"],
					'algorithm': ['auto'],#'ball_tree','kd_tree','brute'],
					'leaf_size': [1,5,10],#,30],
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


######################
# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self



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



PL_selector()