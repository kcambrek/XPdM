# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:52:42 2019

@author: kees.brekelmans
"""

import sys; sys.executable
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import support
import shap
import shap2
import matplotlib.pyplot as plt
import altair as alt
import sys,os


def run(data, upper_model, model, lower_model, explainer):
    #define relative path
    path = os.path.dirname(os.path.abspath(__file__))

    
    df = data
    #load engine ID that has been used in the monitor interface
    with open(path + "/engine", 'rb') as handle:
            engine = pickle.load(handle)
    #select data from specific engine
    engine_df = df[df['unit'] == engine].drop(columns = ["unit", "RUL"]).reset_index(drop = True)
    #retrieve preditions
    upper, Y, lower = upper_model.predict(engine_df).round(1), model.predict(engine_df).round(1), lower_model.predict(engine_df).round(1)
    
    predictions = pd.DataFrame({"Real trajectory" : list(reversed(range(engine_df.shape[0]))), "Upper confidence bound" : upper, "Predicted": Y, "Lower confidence bound": lower}, index = range(engine_df.shape[0]))
    
    #find most similar historic failure
    most_similar, _ = support.find_most_similar(predictions["Predicted"])
    most_similar = int(most_similar)
    most_similar_RUL = support.load_historic_RUL()[most_similar]

    predictions["Engine " + str(int(most_similar))] = most_similar_RUL[:predictions.shape[0]]

    #interfacce instructions            
    st.write("In the diagnose mode, you can perform a thorough analysis on the engine you selected in the monitor mode. The application provides you with a historic failure which is the most similar to the failure of the selected engine. Further, you can look back at each cycle to see which feature and engine part contributed most to the prediction.")
    
    #explanation candidate message
    st.info("Failure seems to be most similar to historic failure in engine " + str(most_similar))
    
    #build main plot
    base = alt.Chart(predictions.reset_index(), height=500).encode(alt.X("index:Q", scale = alt.Scale(domain = (0, 350)), title = "Cycle"), alt.Y(scale = alt.Scale(domain = (0,350)), title = "Remaining predicted cycles"))
    
    show_true = st.checkbox("Show True remaining cycles.")
    show_most_similar = st.checkbox('Show historic predictions of' +" engine " + str(most_similar))

    z1 = base.mark_line(color = 'blue').encode(y='Predicted:Q')
    z2 = base.mark_area(opacity=0.3).encode(
        y='Upper confidence bound:Q',
        y2='Lower confidence bound:Q'
    )
   
    plot =  z1 + z2
    
    if show_true:
        z3 = base.mark_line(color='red').encode(y='Real trajectory:Q')
        plot += z3
    if show_most_similar:
        z4 = base.mark_line(color='orange').encode(y="Engine " + str(int(most_similar))+ ":Q")
        plot += z4
    
    plot.layer[0].encoding.y.title = 'Remaining predicted cycles'
    st.altair_chart(plot.interactive())     
    
    #create cycle select slider
    cycle = st.slider('Select cycle', 0, engine_df.shape[0]-1)
    
    #display predictions at selected cycle
    low = st.subheader("Lower confidence bound: " + str(lower[cycle]))
    pred = st.subheader("Predicted: " + str(Y[cycle]))
    up = st.subheader("Upper confidence bound: " + str(upper[cycle]))
    
    
    
    #select plot to display
    plot_type = st.selectbox("Select visualization", ["Engine-plot", "Decision-plot"])
    
    show_historic = st.checkbox("Compare with historic failure.")
    
    #place holders for plot header and plots
    plot_header = st.empty()
    diagnose_plot = st.empty()
    historic_plot_header = st.empty()
    historic_plot = st.empty()
    
    if show_historic:
        history = support.load_history()
        historic_df = history[history['unit'] == "ID " + str(most_similar)].drop(columns = ["unit", "RUL"]).reset_index(drop = True)
        historic_df["cycle"] = cycle
    
    
    #create decision plot
    if plot_type == "Decision-plot":
        plot_header.subheader("Influence of all measures on prediciton at cycle " + str(cycle) + " of engine " + engine)
        engine_df["cycle"] = cycle
        diagnose_plot.pyplot(shap2.decision_plot(explainer.expected_value, explainer.shap_values(engine_df.iloc[cycle]), engine_df, show = False), bbox_inches = "tight")
        #optional historic plot
        if show_historic:
            plt.clf()
            cycle = min(cycle, historic_df.shape[0]-1)
            historic_plot_header.subheader("Influence of all measures on prediciton at cycle " + str(cycle) + " of engine " + "ID " + str(most_similar))
            historic_plot.pyplot(shap2.decision_plot(explainer.expected_value, explainer.shap_values(historic_df.iloc[cycle]), historic_df, show = False), bbox_inches = "tight")
    
    #create engine plot
    if plot_type == "Engine-plot":  
        SHAP_values = pd.DataFrame(explainer.shap_values(engine_df.values[cycle].reshape((1, engine_df.shape[1]))), columns = engine_df.columns)
        
            
        plot_header.subheader("Influence of parts on prediciton at cycle " + str(cycle) + " of engine " + engine)
        diagnose_plot.pyplot(support.create_engine_plot(SHAP_values), bbox_inches = "tight")
        #optional historic plot
        if show_historic:
            cycle = min(cycle, historic_df.shape[0]-1)
            historic_SHAP_values = pd.DataFrame(explainer.shap_values(historic_df.values[cycle].reshape((1, historic_df.shape[1]))), columns = historic_df.columns)
            historic_plot_header.subheader("Influence of parts on prediciton at cycle " + str(cycle) + " of engine " + "ID " + str(most_similar))
            plt.axis('off')
            historic_plot.pyplot(support.create_engine_plot(historic_SHAP_values), bbox_inches = "tight")

    
    
    
    
    
    
    