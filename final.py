import tkinter
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import *

import numpy as np 
import pandas as pd 


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import cv2
import mahotas as mt

import warnings
warnings.filterwarnings('ignore')

main = tkinter.Tk()
main.title("Blood Cell classification")
main.geometry("1300x1200")

class test:
	def upload():
		global filename
		text.delete('1.0', END)
		filename = askopenfilename(initialdir = "Dataset")
		pathlabel.config(text=filename)
		text.insert(END,"Dataset loaded\n\n")

	def csv():
		global data
		text.delete('1.0', END)
		data=pd.read_csv(filename)
		text.insert(END,"Top Five rows of dataset\n"+str(data.head())+"\n")
		text.insert(END,"Last Five rows of dataset\n"+str(data.tail()))
		data.drop('Unnamed: 0',axis=1,inplace=True)

		
	def splitdataset():
                
	    text.delete('1.0', END)
	    #print(data.columns)
	    X = data.iloc[:,:-1] 
	    Y = data.iloc[:,-1]
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 15)
	    text.insert(END,"\nTrain & Test Model Generated\n\n")
	    text.insert(END,"Total Dataset Size : "+str(len(data))+"\n")
	    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
	    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")
	    return X_train, X_test, y_train, y_test

	def MLmodels():
            global model_final
            X_train, X_test, y_train, y_test=test.splitdataset()
            text.delete('1.0', END)
            models=[]
            models.append(('RandomForest',RandomForestClassifier()))
            models.append(('DecisionTree',DecisionTreeClassifier()))
            models.append(('Adaboost',AdaBoostClassifier()))
            models.append(('Bagging',BaggingClassifier()))
            results=[]
            names=[] 
            predicted_values=[]
            text.insert(END,"Machine Learning Classification Models\n")
            text.insert(END,"Predicted values,Accuracy Scores and S.D values from ML Classifiers\n\n")
            for name,model in models:
                    kfold=KFold(n_splits=10,random_state=7)
                    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
                    model.fit(X_train,y_train)
                    predicted=model.predict(X_test)

                    predicted_values.append(predicted)
                    results.append(cv_results.mean()*100)
                    names.append(name)
                    text.insert(END,"\n"+str(name)+" "+"Predicted Values on Test Data:"+str(predicted)+"\n\n")
                    text.insert(END, "%s: %f\t\t(%f)\n" %(name,cv_results.mean()*100,cv_results.std()))
                    if name == 'Bagging':
                        model_final=model
            return results
	        
	def graph():
	    results=test.MLmodels()	    
	    bars = ('RandomForest','DecisionTree','Adaboost','Bagging')
	    y_pos = np.arange(len(bars))
	    plt.bar(y_pos, results)
	    plt.xticks(y_pos, bars)
	    plt.show()
	def singleImage():
            global testfile
            main_img = cv2.imread(testfile)
            #Preprocessing
            img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
            gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gs, (25,25),0)
            ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            kernel = np.ones((50,50),np.uint8)
            closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

            #Shape features
            contours, image = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            rectangularity = w*h/area
            circularity = ((perimeter)**2)/area

            #Color features
            red_channel = img[:,:,0]
            green_channel = img[:,:,1]
            blue_channel = img[:,:,2]
            blue_channel[blue_channel == 255] = 0
            green_channel[green_channel == 255] = 0
            red_channel[red_channel == 255] = 0

            red_mean = np.mean(red_channel)
            green_mean = np.mean(green_channel)
            blue_mean = np.mean(blue_channel)

            red_std = np.std(red_channel)
            green_std = np.std(green_channel)
            blue_std = np.std(blue_channel)

            #Texture features
            textures = mt.features.haralick(gs)
            ht_mean = textures.mean(axis=0)
            contrast = ht_mean[1]
            correlation = ht_mean[2]
            inverse_diff_moments = ht_mean[4]
            entropy = ht_mean[8]
            vector = [area,perimeter,w,h,aspect_ratio,rectangularity,circularity, red_mean,green_mean,blue_mean,red_std,green_std,blue_std,
                      contrast,correlation,inverse_diff_moments,entropy]
            return vector

	def pred():
                global model_final
                global testfile
                text.delete('1.0', END)
                testfile = askopenfilename(initialdir = "Dataset")
                text.insert(END,"Predict File Selected\n\n")
                vector = test.singleImage()
                test_data = pd.DataFrame([vector])
                               
                pred=model_final.predict(test_data)
                classess=['Lymphocyte','Monocyte','Neutrophil','Eosinophil']
                print(pred)
                text.insert(END,"For Image "+str(filename)+" Predicted Output is "+str(classess[pred[0]-1])+"\n")

font = ('times', 16, 'bold')
title = Label(main, text='Image Classification of Abnormal Red Blood Cells Using Decision Tree Algorithm')
title.config(bg='sky blue', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=test.upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='royal blue', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

df = Button(main, text="Reading Data ", command=test.csv)
df.place(x=700,y=200)
df.config(font=font1)

split = Button(main, text="Train_Test_Split ", command=test.splitdataset)
split.place(x=700,y=250)
split.config(font=font1)

ml= Button(main, text="All Classifiers", command=test.MLmodels)
ml.place(x=700,y=300)
ml.config(font=font1) 

graph= Button(main, text="Model Comparison", command=test.graph)
graph.place(x=700,y=350)
graph.config(font=font1)

pre= Button(main, text="Predict", command=test.pred)
pre.place(x=700,y=400)
pre.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='powder blue')
main.mainloop()
