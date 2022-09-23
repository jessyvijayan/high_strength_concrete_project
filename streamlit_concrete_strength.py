import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.cluster import KMeans
  
# loading in the model to predict on the data
pickle_in = open('concrete_strength.pkl', 'rb')
classifier = pickle.load(pickle_in)

df = pd.read_excel('concrete_strength_data.xlsx')


def prediction(cement,slag,ash,water,superplastic,coarseagg,fineagg,age):  
    
    if age <= 1 & age <= 3:
        age_bins = 1
    elif age <= 4 & age <= 14:
        age_bins = 2
    elif age <= 15 & age <= 28:
        age_bins = 3
    elif age <= 29 & age <= 90:
        age_bins = 4
    else:
        age_bins = 5
    
    data = {'cement' : [cement],
            'slag' : [slag],
            'ash':[ash],
            'water':[water],
            'superplastic':[superplastic],
            'coarseagg':[coarseagg],
            'fineagg':[fineagg],
            'age_bins' :[age_bins]}
    
    new_df = pd.DataFrame(data)
    labels = KMeans(n_clusters = 2,random_state = 10)
    cluster = labels.fit_predict(df.drop('strength',axis=1))
    complete_df = df.join(pd.DataFrame(cluster,columns=['clusters']))
    temp_df = complete_df.groupby('clusters')['cement'].agg(['mean','median'])
    
    new_cluster = labels.predict(new_df)   

    df1 = new_df.join(pd.DataFrame(new_cluster,columns=['clusters']))
    cluster_df = df1.merge(temp_df,on= 'clusters',how='left')


    pred = classifier.predict(cluster_df) 

    return pred[0]
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Determination of the strength of high-performance concrete")
          
    # the following lines create text boxes in which the user can enter the data required to make the prediction
    cement = st.number_input("Cement Quantity(kg in m^3 mixture)")
    slag = st.number_input("Slag Quantity(kg in m^3 mixture)")
    ash = st.number_input("Fly Ash Quantity(kg in m^3 mixture)")
    water = st.number_input("Amount of Water(kg in m^3 mixture)")
    superplastic = st.number_input("Superplastic Quantity(kg in m^3 mixture)")
    coarseagg = st.number_input("Coarse Aggregate Quantity(kg in m^3 mixture)")
    fineagg = st.number_input("Fine Aggregate Quantity(kg in m^3 mixture)")
    age = st.number_input("Age at Testing of Concrete(days:1-365)")

    result =''
    
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(cement,slag,ash,water,superplastic,coarseagg,fineagg,age) 
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()