# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:15:54 2019

@author: kees.brekelmans
"""

import pickle
import streamlit as st
import sklearn
import numpy as np
import pandas as pd
import joblib


@st.cache
def load_linear_model():
    model = joblib.load(r"jupyter\model")
    return(model)
    
#def load_gb_model():
#    lower_model = pickle.load(open(r"C:\Users\kees.brekelmans\python\Thesis\gb_lower_model", "rb"))
#    print(sklearn.utils.validation.check_is_fitted(lower_model, "coef_"))
#    model = pickle.load(open(r"C:\Users\kees.brekelmans\python\Thesis\gb_model", "rb"))
#    upper_model = pickle.load(open(r"C:\Users\kees.brekelmans\python\Thesis\gb_upper_model", "rb"))
#    return lower_model, model, upper_model
    
@st.cache 
def load_gb_model():
    lower_model = joblib.load(r"C:\Users\kees.brekelmans\python\Thesis\gb_lower_model")
    #print(sklearn.utils.validation.check_is_fitted(lower_model, "coef_"))
    model = joblib.load(r"C:\Users\kees.brekelmans\python\Thesis\gb_model")
    upper_model = joblib.load(r"C:\Users\kees.brekelmans\python\Thesis\gb_upper_model")
    return lower_model, model, upper_model
   
    
def predict(x):
    model = load_linear_model()
    print(x)
    prediction = model.predict(x)
    return prediction - 5, prediction, prediction +5

#def predict(x):
#    #model = load_model()
#    lower_model, model, upper_model = load_gb_model()
#    lower_prediction = lower_model.predict(x)
#    prediction = model.predict(x)
#    upper_prediction = upper_model.predict(x)
#    #return(prediction - 5, prediction, prediction +5)
#    return lower_prediction, prediction, upper_prediction