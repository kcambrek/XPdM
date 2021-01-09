# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:41:26 2019

@author: kees.brekelmans
"""
import sys; sys.executable
import streamlit as st
import pandas as pd
import numpy as np 
import pickle
import time
import shap
import shap2
import matplotlib.pyplot as plt
from PIL import Image
import support
import altair as alt
import os


def run(data, upper_model, model, lower_model, explainer):
    path = os.path.dirname(os.path.abspath(__file__))
    
    #load data set with only input features
    #df = data.drop(columns = ["unit", "RUL"])
    df = support.load_history().drop(columns = ["unit", "RUL"])
    st.write("In the What-if mode, you can tweek input features and explore how these impact the remaining engine cycles according to the predictive model. You can select different visualizations to support you in your exploration.")
    
    
    #creates plot selection tool    
    plot_type = st.selectbox("Select visualizations", ["Engine-plot", "Decision-plot", "Measure-plot"])
    measure = st.empty()
    plot_header = st.empty()
    plot = st.empty()
        

    def map_sensors(df):
        #creates dict that stores the min, mean, max and std for each feature to create the sliders
        sensors = {sensor: {"min" : df[sensor].min(), "start" : df[sensor].mean(), "max" : df[sensor].max(), "std" : df[sensor].std()} for sensor in df.columns}
        return sensors
    sensors = map_sensors(df)
    
    #creates for all the features a slider with a minimum and maximum, subtracted and added by two times the std respectively.   
    for sensor in sensors.keys():
        #Treat cycle differently to prevent negative number of cycles.
        if sensor == "cycle":
            sensors[sensor]["start"] = st.slider(sensor, 0.0, float(sensors[sensor]["max"] + sensors[sensor]["std"]*2), float(sensors[sensor]["start"]))
        else:  
            sensors[sensor]["start"] = st.slider(sensor, float(sensors[sensor]["min"] - sensors[sensor]["std"]*2), float(sensors[sensor]["max"] + sensors[sensor]["std"]*2), float(sensors[sensor]["start"]))
    
    #create empty dataframe with the column names from the original input data
    df_ = pd.DataFrame(index=[0], columns=df.columns)
    #fill the empty dataframe with the values returned by the sliders
    for sensor in sensors.keys():
        df_[sensor][0] = sensors[sensor]["start"]
        
    #displays predictions
    st.sidebar.subheader("Lower confidence bound: " + str(lower_model.predict(df_).round(1)[0]))
    st.sidebar.subheader("Predicted: " + str(model.predict(df_).round(1)[0]))
    st.sidebar.subheader("Upper confidence bound: " + str(upper_model.predict(df_).round(1)[0]))
    
    
    #creates measure plot
    if plot_type == "Measure-plot":
        #try to reload previous selected meausure, since streamlit resets all the variables when new values are selected
        try:
            measure_index = pickle.load(open(path + "/measure", "rb"))
        except:
            measure_index = 0
        #select measure to plot in the measure plot
        measure = measure.selectbox("Select measures to monitor", df.columns, index = measure_index)
        measure_index = list(df.columns).index(measure)
        pickle.dump(measure_index, open(path + "/measure", "wb"))
        plot_header.header("Remaining predicted cycles vs {} under current conditions".format(measure))
        repititions = 100
        df_measure = pd.concat([df_]*repititions, ignore_index=True)
        #cycle is a special case, since we do not want to plot negative cycles
        if measure == "cycle":
            df_measure[measure] = np.linspace(0, sensors[measure]["max"] + sensors[measure]["std"]*2, repititions)
        else:
            df_measure[measure] = np.linspace(sensors[measure]["min"] - sensors[measure]["std"]*2, sensors[measure]["max"] + sensors[measure]["std"]*2, repititions)
        predicted = model.predict(df_measure).round(1)
        upper = upper_model.predict(df_measure).round(1)
        lower = lower_model.predict(df_measure).round(1)
        
        df_measure["predicted"] = predicted
        df_measure["upper"] = upper
        df_measure["lower"] = lower
        df_measure["in_train_range"] = ["yes" if sensors[measure]["min"] <= i <= sensors[measure]["max"] else "before" if i < sensors[measure]["min"] else "after" for i in df_measure[measure]]
        
        #build measure plot
        base = alt.Chart(df_measure.reset_index(), height=500).encode(alt.X(measure, title = measure), alt.Y(scale = alt.Scale(domain = (0,350)), title = "Remaining cycles"))
        
        
        domain = ['yes', 'before', 'after']
        range_ = ['blue', 'orange', 'orange']
        

                    
        z1 = base.mark_line(color='blue').encode(y='predicted:Q', color = alt.Color("in_train_range:N",legend=None, scale=alt.Scale(domain=domain, range=range_)))
        z2 = base.mark_area(opacity=0.3).encode(
            y='upper:Q',
            y2='lower:Q',
            color = alt.Color("in_train_range:N", legend=alt.Legend(title="In training set"))
        )
                       
        alt_plot = z1 + z2
        alt_plot.layer[0].encoding.y.title = 'Remaining predicted cycles'

        plot.altair_chart(alt_plot.interactive())

    #create decision plot
    if plot_type == "Decision-plot":
        plot_header.header("Contributions of features to the prediciton")
        plot.pyplot(shap2.decision_plot(explainer.expected_value, explainer.shap_values(df_), df_, show = False),  bbox_inches = "tight")
    
    #create engine plot
    if plot_type == "Engine-plot": 
        plot_header.header("Contributions of engine parts to the prediction")
        SHAP_values = pd.DataFrame(explainer.shap_values(df_), columns = df_.columns)
        plot.pyplot(support.create_engine_plot(SHAP_values), bbox_inches = "tight")
        
        
    
    
    