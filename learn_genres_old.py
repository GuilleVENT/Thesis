## used in init 
from os import listdir
from glob import glob
import pandas
import numpy as np

## used for kfold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

## used for learn
#from sklearn.cross_validation import ShuffleSplit
#from sklearn.linear_model.logistic import LogisticRegression

## used for training
from sklearn.svm import SVC

## used for parameter selection
from sklearn import svm
from sklearn.model_selection import GridSearchCV

## used prediction and outputs
import heapq
import csv


## debugging/develop
import sys



#global
genre_list = []

headers = [['ID', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']]
#top3 = [['ID','genre1','genre2','genre3']]

def init():

	path_genreDB =  r"genre_outputs/"

	#path_array = glob(path_genreDB+"*/")
	'''
	print path_array
	for elem in path_array:
		#print elem[76:][:-1]
		genre_list.append(elem[76:][:-1])
	print genre_list
	'''
	
	y_list_num = []
	y_list 	   = []
	X_list 	   = []

	# delete blues and country
	#path_array.pop(0)	# poping blues
	#print path_array
	#path_array.pop(1)	# poping country
	#print path_array

	for genre in listdir(path_genreDB):
		if genre.startswith('.'):
			pass
		else:
			genre_list.append(genre)
	
	print(genre_list)
	#sys.exit('stop')
	
	for path_genre in listdir(path_genreDB):
		path_ = path_genreDB+path_genre+"/"

		X_list_genre = []

		for file in listdir(path_):
			if file.startswith('OUT'):			# from Output. Matlab outputs
				print(file)
				file_path = path_ +file
				
				#sys.exit(file_path)

				dataframe = pandas.read_csv(file_path)
				#print dataframe
				
				dataframe = dataframe.drop('Unnamed: 36',1)		# dropping unnamed 
				

				dataframe = dataframe.drop(dataframe.columns[33], axis=1) 		# dropping duplicate column by index = 33 	
			

				for feature in dataframe.columns:
					if feature.count('.')>1:
						feature_n = feature[:-2]
						dataframe.rename(columns={feature: feature_n}, inplace=True)					# eliminating dobble dots (actually more like renaming the column)
						
	
				#print dataframe
				df_list = list(dataframe)
				df_list = [float(i) for i in list(dataframe)]
				#print df_list
				
				'''		#SELECTION: testing to get the right one. 
				print "ASR ", df_list[19]		
				print "Brightness m+std ", df_list[3], " ", df_list[4] 
				print "Cepstrum m+std ", df_list[27], " ", df_list[28]
				print "Event_density ", df_list[24]
				print "mfcc m+std ", df_list[20],' ',df_list[21]
				print "onsets std ", df_list[14]
				print "pulse_clarity ", df_list[12]
				print "RMS m+std", df_list[8], ' ',df_list[9]
				print "Rolloff85 m+std", df_list[1],' ',df_list[2]
				print "Roughness ", df_list[25]
				print "SpectralFlux m+std ", df_list[5], ' ' ,df_list[6]
				print "Tempo ", df_list[11]
				print "ZeroCross ", df_list[7] 
				'''
				
				df_list_selected = [df_list[19],df_list[3],df_list[4],df_list[27],df_list[28],df_list[24],df_list[20],df_list[21],df_list[14],df_list[12],df_list[8],df_list[9],df_list[1],df_list[2],df_list[25],df_list[5],df_list[6],df_list[11],df_list[7]]
				print(df_list_selected)


				#print df_list

				X_list_genre.append(df_list_selected)

				df_list_np = np.array(df_list_selected)

				print(df_list_np) 

				'''
				print df_list_np.dtype
				print type(df_list_np)
				print type(df_list_np[1])
				print df_list_np.shape
				print " RESHAPING"
				'''
				df_reshaped = np.reshape(df_list_np,(1,19))		#FROM (1,35) TO (1,19) BECAUSE OF SELECTION 
				print(df_reshaped.shape)

				if X_list == []:
					X_list = df_reshaped

				else:
					X_list = np.vstack((X_list,df_reshaped))
			
		print(X_list)
		
		print(X_list.shape)
		print(X_list.dtype)
		print("TYPE X       :" , type(X_list))
		print("TYPE X[0]    :" , type(X_list[0]))
		print("TYPE X[0][0] :" , X_list[0][0] , " " ,type(X_list[0][0])) 
	
		y_list_num.append(len(X_list_genre))
		print(y_list_num)

	for index, value in enumerate(y_list_num):
		y_list_temp = [index] * value
		y_list.extend(y_list_temp)

	y_list = np.array(y_list)
	print(y_list)
	print(X_list)
	print(X_list.shape)
	print(X_list.dtype)


	k_fold(X_list, y_list)
	#shuffle_split(X_list,y_list)


def k_fold(X,y):
	## number of folds
	num_folds = 5 

	kf = KFold(n_splits = num_folds,shuffle=True)
	kf.get_n_splits(X)

	print(kf) 

	clfs = []  # for best selection

	cms = []	# multiple confussion matrixes
	
	scores = [] 	# for the results
	train_errors = []
	test_errors = []	

	## select parameters:
	C_value , gamma_value = 10000,1e-8 #svc_param_selection(X, y, num_folds)
	#SVC_param(X,y,num_folds)

	##

	
	for train_index, test_index in kf.split(X):
		
		print("TRAIN: ")
		print(train_index)
		print("TEST:  ")
		print(test_index)

		
		X_train = X[train_index]
		print(type(train_index))
		print(train_index.dtype)
		#sys.exit('stop')
		X_test	= X[test_index]
		
		y_train = y[train_index]
		y_test  = y[test_index]

		print("X_train")
		print(X_train)
		print("y_train")
		print(y_train)
		print("X_test")
		print(X_test)
		print("y_test")
		print(y_test)

		#clf = SVC(kernel='poly',degree=2)
		clf = SVC(C=C_value,kernel='rbf',gamma=gamma_value, probability=True)
		
		clf.fit(X_train, y_train)

		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)


		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		clfs.append(clf)

		cms.append(cm)

	print(" - - - - - - - - - - - ")
	print("CONFUSSION MATRIX:")
	print(np.asarray(cms))
	print(" - - - - - - - - - - - ")
	print("TRAIN ERRORS: ") 
	print("mean ", np.mean(train_errors))
	print(train_errors)
	print(" - - - - - - - - - - - ") 
	print("TEST  ERRORS: ")  
	print("mean ",np.mean(test_errors))
	print(test_errors)
	print(" - - - - - - - - - - - ")
	print("TRAINING METHOD")
	print("kernel: 'rbf'") 
	print("folds : ", num_folds)
	print("C     : ", C_value)
	print("gamma : ", gamma_value)
	print(" - - - - - - - - - - - ")

	# choose clf with the least TEST errors 
	print("Choosing Classifier:")
	print("Test Errors: ",min(test_errors))
	#print test_errors.index(min(test_errors))
	clf_def = clfs[test_errors.index(min(test_errors))]
	if min(test_errors)>0.40:
		recall(X,y)

	print(" ")
	print(" - - - - END - - - - -")
	print(" - - - - - - - - - - - ")

	# pass clf_def to next function 


	#get_outputs(clf_def)

def recall(X,y):
	print(" Restarting Learning Process ")
	k_fold(X,y)


def get_outputs(clf):

	path = '/Users/guillermoventuramartinezAIR/Desktop/develop/python2matlab/'

	users_l = glob(path+'*/')

	for user in users_l:
		user_pl = glob(user+'*/')
		
		for pl in user_pl:
			file_l = listdir(pl)
			print(pl)

			for file in file_l:
				if file.startswith('OUT'):
					print(" - - - - - - - - - - - - - - - - -")
					print(file)

					file_path = pl+file

					dataframe = pandas.read_csv(file_path)
					#print dataframe
					
					dataframe = dataframe.drop('Unnamed: 36',1)		# dropping unnamed 
					

					dataframe = dataframe.drop(dataframe.columns[33], axis=1) 		# dropping duplicate column by index = 33 	
				

					for feature in dataframe.columns:
						if feature.count('.')>1:
							feature_n = feature[:-2]
							dataframe.rename(columns={feature: feature_n}, inplace=True)					# eliminating dobble dots (actually more like renaming the column)
							
		
					#print dataframe
					df_list = list(dataframe)
					df_list = [float(i) for i in list(dataframe)]
					#print df_list
					'''
					#SELECTION: testing to get the right one. 
					print "ASR ", df_list[19]		
					print "Brightness m+std ", df_list[3], " ", df_list[4] 
					print "Cepstrum m+std ", df_list[27], " ", df_list[28]
					print "Event_density ", df_list[24]
					print "mfcc m+std ", df_list[20],' ',df_list[21]
					print "onsets std ", df_list[14]
					print "pulse_clarity ", df_list[12]
					print "RMS m+std", df_list[8], ' ',df_list[9]
					print "Rolloff85 m+std", df_list[1],' ',df_list[2]
					print "Roughness ", df_list[25]
					print "SpectralFlux m+std ", df_list[5], ' ' ,df_list[6]
					print "Tempo ", df_list[11]
					print "ZeroCross ", df_list[7] 
					'''
					
					df_list_selected = [df_list[19],df_list[3],df_list[4],df_list[27],df_list[28],df_list[24],df_list[20],df_list[21],df_list[14],df_list[12],df_list[8],df_list[9],df_list[1],df_list[2],df_list[25],df_list[5],df_list[6],df_list[11],df_list[7]]
					#print df_list_selected


					#sys.exit('developing')

					df_list_np = np.array(df_list_selected)

					print(df_list_np) 

					'''
					print df_list_np.dtype
					print type(df_list_np)
					print type(df_list_np[1])
					print df_list_np.shape
					print " RESHAPING"
					'''
					df_reshaped = np.reshape(df_list_np,(1,19))		#FROM (1,35) TO (1,19) BECAUSE OF SELECTION 
					print(df_reshaped.shape)

					X_song  = df_reshaped
					#print "X_song:"
					#print X_song									# this pandas work and listing and reshaping could be all one callable function inside this program. It's called twice. 

					prediction_2(clf,X_song,file)
					#prediction(clf,X_song,file)
					# this predicts one for one the genre of each song.
	# then the output csv should be written
	output_2()
	#output()

						
def prediction(clf,X_song,file):
	print(" Predicting........")
	print("SONG ID:      " + file[7:][:-4])
	probs = clf.predict_proba(X_song.reshape(1,-1))
	probs=probs[0]
	print(probs)

	top3_genre = []
	top3_porct = []
	top3_genre = heapq.nlargest(3, range(len(probs)), key=probs.__getitem__)	
	top3_porct = heapq.nlargest(3, probs)

	top3_porct[0] = top3_porct[0] * 100		# making it porcentages from 0 to 100 
	top3_porct[1] = top3_porct[1] * 100		# making it porcentages from 0 to 100 
	top3_porct[2] = top3_porct[2] * 100 	# making it porcentages from 0 to 100 
											# this could be inside the while loop down here
	i = 0 
	while i < 3 :
		#print top3_genre[i]
		temp = genre_list[top3_genre[i]]
		top3_genre[i] = temp
		print(top3_genre[i])
		i  = i+1
		#print genre_list[top3_genre[i]]
	
	print(top3_genre)
	print(top3_porct)

	tuple_temp_0 = (top3_genre[0],str(top3_porct[0]))
	tuple_temp_1 = (top3_genre[1],str(top3_porct[1]))
	tuple_temp_2 = (top3_genre[2],str(top3_porct[2]))

	top3.append([file[7:][:-4],tuple_temp_0,tuple_temp_1,tuple_temp_2])


def prediction_2(clf,X_song,file):
	print("Predicting (ver2) ............")
	print("SONG_ID:		"+ file[7:][:-4])
	probs = clf.predict_proba(X_song.reshape(1,-1))
	probs=probs[0]
	print(probs)
	#print list(probs) 
	#print genre_list
	
	genres_res = []


	i = 0 
	while i < len(genre_list):
		tuple_temp = (genre_list[i],probs[i])
		genres_res.append(tuple_temp)
		i=i+1

	#print genres_res

	row = [file[7:][:-4],genres_res[0][1],genres_res[1][1],genres_res[2][1],genres_res[3][1],genres_res[4][1],genres_res[5][1],genres_res[6][1],genres_res[7][1],genres_res[8][1]]
	#print row

	headers.append(row)



def output_2():

	path_output = "/Users/guillermoventuramartinezAIR/Desktop/develop/analisis/ID_all_genres.csv"

	with open(path_output, "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for elem in headers:
			writer.writerow(elem)








'''
def shuffle_split(X, y):

	split = ShuffleSplit(n=len(X), n_iter=10, test_size=.1, random_state=0)		# index split in train, test

	clfs = []  # for the median

	cms = []	# multiple confussion matrixes
	
	scores = [] 	# for the results
	train_errors = []
	test_errors = []	

	for train_index , test_index in split:
		print "TRAIN: "
		print train_index
		print "TEST:  "
		print test_index

		
		X_train = X[train_index]
		X_test	= X[test_index]
		
		y_train = y[train_index]
		y_test  = y[test_index]
		
		print "X_train"
		print X_train
		print "y_train"
		print y_train
		print "X_test"
		print X_test
		print "y_test"
		print y_test
		

		#clf = SVC()
		clf = LogisticRegression()
		clf.fit(X_train,y_train)
		
		train_score = clf.score(X_train, y_train)
		test_score = clf.score(X_test, y_test)

		scores.append(test_score)

		train_errors.append(1 - train_score)
		test_errors.append(1 - test_score)


		y_pred = clf.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)

		cms.append(cm)

	print " - - - - - - - - - - - "
	print "CONFUSSION MATRIX:"
	print np.asarray(cms)
	print " - - - - - - - - - - - "
	print "TRAIN ERRORS: " 
	print np.mean(train_errors)
	print " - - - - - - - - - - - " 
	print "TEST  ERRORS: "  
	print np.mean(test_errors)
	print " - - - - - - - - - - - "
'''

def svc_param_selection(X, y, nfolds):
	print(" CALCULATIN C and GAMMA in KERNEL RBF")

	Cs = [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]
	gammas = [1e-8,1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000]
	param_grid = {'C': Cs, 'gamma' : gammas}

	grid_search_ = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
	grid_search_.fit(X, y)
	grid_search_.best_params_
	#return grid_search_.best_params_
	print(grid_search_.best_params_)
	print(type(grid_search_.best_params_))
	print("C ",grid_search_.best_params_['C'])
	print("GAMMA " ,grid_search_.best_params_['gamma'])
	
	return grid_search_.best_params_['C'],grid_search_.best_params_['gamma']
	
	#debugging vvv (faster)
	#return 10000, 1e-8

'''
def SVC_param(X,y,nfolds):

	Cs = [1,2,3,4,5,6,7,8,9,10,100,1000,10000]
	gammas = [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]

	parameters = {'kernel':('rbf','linear','sigmoid'),'C':Cs,'gamma':gammas}

	clf = SVC()

	grid = grid_search.GridSearchCV(clf, parameters)

	grid.fit(X,y)

	print grid.best_params_
	sys.exit('STOP pls')
'''


#developing
#get_outputs('test')
init()