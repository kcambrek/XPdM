# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:51:20 2019

@author: kees.brekelmans
"""
import pickle
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

#define relative path
path = os.path.dirname(os.path.abspath(__file__))

@st.cache
def load_data():
    #loads and returns the demo data containing the sensory data for 5 engines
    df = pickle.load(open(path + "/data/demo_df", "rb"))

    df["unit"] =  "ID " + df['unit'].astype(int).astype(str)
    return df

    
@st.cache 
def load_gb_model():
    #loads and returns the three saved models to predict RUL, including upper and lower prediction confidence bounds
    lower_model = joblib.load(path + "/jupyter\gb_lower_model.joblib")
    model = joblib.load(path + "/jupyter\gb_model.joblib")
    upper_model = joblib.load(path + "/jupyter\gb_upper_model.joblib")
    
    return  lower_model, model, upper_model
   

@st.cache
def load_image(file_name):
    return plt.imread(file_name)


def sensors_to_parts(SHAP_values):
    #maps the sensory data to the parts by aggregating the part-relevant sensors on positive and negative Shapley-values seperately        
    SHAP_values["cycle_neg"] = SHAP_values[["cycle"]][SHAP_values[["cycle"]] < 0].sum(axis = 1)
    SHAP_values["Fan_neg"] = SHAP_values[["Physical fan speed", "Physical core speed", "Corrected fan speed", "Bleed Enthalpy"]][SHAP_values[["Physical fan speed", "Physical core speed", "Corrected fan speed", "Bleed Enthalpy"]] < 0].sum(axis = 1)
    SHAP_values["High-Pressure Turbine_neg"] = SHAP_values[["HPT coolant bleed"]][SHAP_values[["HPT coolant bleed"]] < 0].sum(axis = 1)
    SHAP_values["High-Pressure Compressor_neg"] = SHAP_values[["Total temperature at HPC outlet", "Total pressure at HPC outlet", "Static pressure at HPC outlet", "Ratio of fuel flow to Ps30"]][SHAP_values[["Total temperature at HPC outlet", "Total pressure at HPC outlet", "Static pressure at HPC outlet", "Ratio of fuel flow to Ps30"]] < 0].sum(axis = 1)
    SHAP_values["Low-Pressure Turbine_neg"] = SHAP_values[["LPT coolant bleed", "Total temperature at LPT outlet"]][SHAP_values[["LPT coolant bleed", "Total temperature at LPT outlet"]] < 0].sum(axis = 1)
    SHAP_values["Low-Pressure Compressor_neg"] = SHAP_values[["Total temperature at LPC outlet"]][SHAP_values[["Total temperature at LPC outlet"]] < 0].sum(axis = 1)
    SHAP_values["Nozzle_neg"] = SHAP_values[["Total pressure in bypass-duct", "Bypass Ratio"]][SHAP_values[["Total pressure in bypass-duct", "Bypass Ratio"]] < 0].sum(axis = 1)
    
    SHAP_values["cycle_pos"] = SHAP_values[["cycle"]][SHAP_values[["cycle"]] > 0].sum(axis = 1)
    SHAP_values["Fan_pos"] = SHAP_values[["Physical fan speed", "Physical core speed", "Corrected fan speed", "Bleed Enthalpy"]][SHAP_values[["Physical fan speed", "Physical core speed", "Corrected fan speed", "Bleed Enthalpy"]] > 0].sum(axis = 1)
    SHAP_values["High-Pressure Turbine_pos"] = SHAP_values[["HPT coolant bleed"]][SHAP_values[["HPT coolant bleed"]] > 0].sum(axis = 1)
    SHAP_values["High-Pressure Compressor_pos"] = SHAP_values[["Total temperature at HPC outlet", "Total pressure at HPC outlet", "Static pressure at HPC outlet", "Ratio of fuel flow to Ps30"]][SHAP_values[["Total temperature at HPC outlet", "Total pressure at HPC outlet", "Static pressure at HPC outlet", "Ratio of fuel flow to Ps30"]] > 0].sum(axis = 1)
    SHAP_values["Low-Pressure Turbine_pos"] = SHAP_values[["LPT coolant bleed", "Total temperature at LPT outlet"]][SHAP_values[["LPT coolant bleed", "Total temperature at LPT outlet"]] > 0].sum(axis = 1)
    SHAP_values["Low-Pressure Compressor_pos"] = SHAP_values[["Total temperature at LPC outlet"]][SHAP_values[["Total temperature at LPC outlet"]] > 0].sum(axis = 1)
    SHAP_values["Nozzle_pos"] = SHAP_values[["Total pressure in bypass-duct", "Bypass Ratio"]][SHAP_values[["Total pressure in bypass-duct", "Bypass Ratio"]] > 0].sum(axis = 1)
    
    return(SHAP_values)
    

def create_engine_plot(SHAP_values, image = path + "/engine.png", scaling = 1000):
    
    #creates the engine plot and returns a matplotlib figure
    SHAP_values = sensors_to_parts(SHAP_values)
    y = 110
    parts = {"cycle" : {"x" : 30, "s": 0}, "Fan" : {"x" : 75, "s" : 0}, "Low-Pressure Compressor" : {"x" : 115, "s" : 0}, "High-Pressure Compressor" : {"x" : 155, "s" : 0}, "Low-Pressure Turbine" : {"x": 250, "s" : 0}, "High-Pressure Turbine" : {"x" : 225, "s" : 0}, "Nozzle" : {"x" : 275, "s" : 0}}
    
    x_engine = []
    y_engine = []
    s_pos_engine = []
    s_neg_engine = []
    
    for part in parts.keys():
        x_engine.append(parts[part]["x"])
        y_engine.append(y)
        s_pos_engine.append(abs(float(SHAP_values[part + "_pos"])*scaling))
        s_neg_engine.append(abs(float(SHAP_values[part + "_neg"])*scaling))
    img = load_image(image)
    fig, ax = plt.subplots(figsize=(20, 20))
    
    ax.scatter(x_engine, [i - 40 for i in y_engine], s = s_pos_engine, alpha = 0.8, color = "green")
    ax.scatter(x_engine, [i + 40 for i in y_engine], s = s_neg_engine, alpha = 0.8, color = "red")
   
    ax.imshow(img)
    
    return fig
    
@st.cache
def load_historic_RUL():
    return(pickle.load(open(path + "/data\historic_RUL", "rb")))
   
@st.cache
def load_history():
    df = pickle.load(open(path + "/data/train_df", "rb"))

    df["unit"] =  "ID " + df['unit'].astype(int).astype(str)
    return df

    
@st.cache    
def find_most_similar(sequence):
    #finds most similar failure from historic data. returns the engine id of the most similar failure and Manhatten distance between found failure and the new failure.
    history = load_historic_RUL()
    max_sequence = find_max_sequence(history)


    sequence = list(sequence) + int(max_sequence - len(list(sequence)))*[0]
    most_similar = 0
    most_similar_distance = 0
    for engine in history.keys():
        distance = sum(abs(np.array(history[engine]) - np.array(sequence)))
        if distance == 0:
            continue
        elif distance < most_similar_distance or most_similar_distance == 0:
            most_similar_distance = distance
            most_similar = engine
    return most_similar, most_similar_distance