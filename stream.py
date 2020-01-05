# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:56:07 2019

@author: kees.brekelmans
"""

import sys; sys.executable
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import monitor
import explore
import diagnose
import support
import shap
import shap2

import os
#define relative path
path = os.path.dirname(os.path.abspath(__file__))

data = support.load_data()

lower_model, model, upper_model = support.load_gb_model()
#get SHAP generator by passing the predictive model
explainer = shap.TreeExplainer(model)

#select box to switch between modes
interface = st.sidebar.selectbox("Choose mode:", ["monitor", "what-if"])

#from monitor mode, one can go to diagnose mode. To keep track of whether the user selected to go to or exit diagnose mode, pickles have to be generated and read, since streamlit runs the whole script from the beginning and all local variables are lost.
if interface == "monitor":
    title = st.empty()
    diagnose_mode = st.button("Go to diagnose mode")
    with open(path + "/diagnose_mode", 'rb') as handle:
        diagnose_mode2 = pickle.load(handle)
    if diagnose_mode == True or diagnose_mode2 == True:
        title.title("Diagnose mode")
        with open(path + "/diagnose_mode", 'wb') as handle:
            pickle.dump(True, handle)
        diagnose_exit = st.button("Ready with diagnose?")
        #the if-else below causes the program to require you to press "Ready with diagnose?" twice. 
        if diagnose_exit:
            with open(path + "/diagnose_mode", 'wb') as handle:
                pickle.dump(False, handle)
        else:
            diagnose.run(data, upper_model, model, lower_model, explainer)
    else:
        title.title("Monitor mode")
        monitor.run(data, upper_model, model, lower_model, explainer)
        diagnose_start = st.button("Diagnose mode")
        if diagnose_start:
            with open(path + "/diagnose_mode", 'wb') as handle:
                pickle.dump(True, handle)
        
#go to what-if mode. Code can be found in explore.py
if interface == "what-if":
    title = st.title("What-if mode")
    explore.run(data, upper_model, model, lower_model, explainer)
        