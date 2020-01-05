# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:05:18 2019

@author: kees.brekelmans
"""
import sys; sys.executable
import streamlit as st
import pandas as pd
#import numpy as np
import pickle
import time
import support
import matplotlib.pyplot as plt
import altair as alt
import sys,os




def run(data, upper_model, model, lower_model, explainer):
    #define relative path
    path = os.path.dirname(os.path.abspath(__file__))
        
    df = data

    st.write("In the monitor mode, you can select engines and measures on the left to monitor their performance. You can set a warning level on the left; when this level falls in the 95% prediction interval, you will be warned of an imminent failure. If you would like to perform a deep dive on the performance of one engine, press the monitor mode button.")    
        
    #engine selection box
    engine = st.sidebar.selectbox('Choose an engine:', df['unit'].unique())
    
    #pickle selected engine to be retrieved in the diagnose mode
    with open(path + "/engine", 'wb') as handle:
            pickle.dump(engine, handle)

    #slider to select warning level
    warning = st.sidebar.slider(
        'Choose RUL warning level:',
         0, 50, 25)
    
    st.header("Engine " + str(engine))
    
    
    #Dummy load data
    #Select only data from selected engine
    engine_df = df[df['unit'] == engine].drop(columns = ["unit"]).reset_index(drop = True)
    
    
    #Remove RUL from predictor variables
    RUL = engine_df.pop("RUL")
    
    #sensory data to be monitored selection box
    measures = st.sidebar.multiselect("Select measures to monitor", engine_df.columns) 
    
    #creates an intial pandas dataframe to be passed to initial plot.
    df1 = pd.DataFrame({"Lower confidence bound": engine_df.count(), "Predicted": engine_df.count(), "Upper confidence bound" : engine_df.count()}, 
                        index = [0])
    
    #place holder for engine status and warning message.
    status = st.empty()
    
    
    #The models first predict everything before the predictions are displayed
    upper, Y, lower = upper_model.predict(engine_df).round(1), model.predict(engine_df).round(1), lower_model.predict(engine_df).round(1)
    
    #Display predictions
    low = st.subheader("Lower confidence bound: " + str(lower[0]))
    pred = st.subheader("Remaining predicted cycles: " + str(Y[0]))
    up = st.subheader("Upper confidence bound: " + str(upper[0]))
    
    #get SHAP values in a dataframe
    SHAP_values = pd.DataFrame(explainer.shap_values(engine_df.values), columns = engine_df.columns)

    #place holder for main plot
    progress_plot = st.empty()
    
    #allows for the display of the engine plot
    monitor_engine = st.checkbox("Show contributions of parts to prediction?")
    #create place holder for engine plot
    if monitor_engine:
        engine_header = st.header("Engine parts and their contributions to the prediction")
        engine_plot = st.empty()
    
    
    def run_engine(engine_df, measures, lower, Y, upper, RUL, monitor_engine, step = 3):
        #iterates through the predictions; updates the displayed predictions, main plot, engine plot and the optional selected measure plots; checks whether the warning should be shown.
        
        #creates and displays the plots of selected sensors
        measure_graphs = []
        for column in measures:
            measure_graphs.append(st.line_chart(engine_df[column][0:1]))
        
        #since rendering the engine plot takes time, the step count is higher, which speeds up the progression
        if monitor_engine:
            step = 10
        
        #iterates through all the predictions and features of the selected engine. in this loop the engine 'runs'
        for cycle, y in enumerate(Y):
            
            #Small hack to retrieve the last cycle, which reflects the time axis and therefore also the index of the current observation.
            pickle.dump(cycle, open(path + "/cycle", "wb"))
            
            #Dataframe which is constantly updated to display all the predicitons untill cycle c. Can probably more efficient.
            df2 = pd.DataFrame({"Upper confidence bound" : upper[:max(1,cycle)], "Predicted": Y[:max(1,cycle)], "Lower confidence bound": lower[:max(1,cycle)], "Warning level" : [warning]*max(1,cycle)}, 
                            index = range(max(1,cycle)))
            
            
            #displays the predictions
            low.subheader("Lower confidence bound: " + str(lower[cycle]))
            pred.subheader("Remaining predicted cycles: " + str(Y[cycle]))
            up.subheader("Upper confidence bound: " + str(upper[cycle]))
            
            
            
            if cycle % step == 0:
                #Creates the main chart
               
                base = alt.Chart(df2.reset_index(), height=500).encode(alt.X("index:Q", scale = alt.Scale(domain = (0, 350)), title = "Cycle"), alt.Y(scale = alt.Scale(domain = (0,350)), title = "Remaining cycles"))

                
                z1 = base.mark_line(color='blue').encode(y='Predicted:Q')
                z2 = base.mark_area(opacity=0.3).encode(
                    y='Upper confidence bound:Q',
                    y2='Lower confidence bound:Q'
                )
                warning_df = pd.DataFrame({"warning_level" : [warning] * 350})
                z3 = base.mark_area(color = "orange", opacity=0.3).encode(
                    y='Warning level:Q'
                )
                               
                plot = z1 + z2 + z3
                plot.layer[0].encoding.y.title = 'Remaining predicted cycles'

                progress_plot.altair_chart(plot.interactive())
                                                    
                
                #Show and updates the selected measures            
                for i, column in enumerate(measures):
                    measure_graphs[i].line_chart((engine_df[column][:cycle]))
                
                if monitor_engine:
                    try:
                        support.create_engine_plot(SHAP_values.iloc[cycle:cycle+1, :])
                        engine_plot.pyplot(plt,  bbox_inches = "tight")
                    except KeyError:
                        pass
                
            
            #Sleep in order to make it look that there is actually something happening.
            time.sleep(0.01)
            
            #Condition to check whether warning should be displayed.
            if lower[cycle] <= warning and RUL[cycle] >= 1:
                status.warning("Engine should be maintained!")
            
            #Condition to check failure and diplay it.
            elif RUL[cycle] < 1:
                status.warning("Engine failed at cycle {}".format(cycle))
                                
                break
            else:
                status.info("Engine is running")

        
        return cycle, df2, progress_plot

    cycle, df2, the_plot = run_engine(engine_df, measures, lower, Y, upper, RUL, monitor_engine)
    

    
