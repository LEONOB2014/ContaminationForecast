'''
File name : Utilites.py
Author: Pablo Camacho Gonzalez
Python version: 3.6.4
Date last modified: 27/02/2018
'''


import numpy as np
from Utilites.oztools import ContIOTools
from sklearn import preprocessing
import re

def converToArray(alldata,contaminant):
    """
    Function that returns in a single column the columns have a specific name

    :param alldata: matrix with all the data of the database
    :type alldata: DataFrame
    :param contaminant: Name of pollutant to look for
    :type contaminant: string
    :return: column with all data of the pollutant
    """
    oztool = ContIOTools()
    name = oztool.findTable(contaminant)
    c = np.ones((len(alldata.index),1));
    columns= alldata.columns;
    patron = re.compile(name+'_.*');
    #patron = re.compile('_*')
    index =0;
    for x in columns:
        if patron.match(x) != None:
            temp = alldata[x];
            for i in temp:
                if np.isnan(i):
                    c[index,0]= -1;
                    index = index +1;
                else:
                    c[index,0]= i;
                    index = index +1;
    return c;


def prepro(data,build,contaminant):
    """
    Function to normalize the data of the neuronal network

    :param data: training data
    :param build: training data
    :param contaminant: name of the contaminant
    :return : normalize data
    """
    x_vals = data.values;
    x = x_vals.shape;
    columns = x[1];
    data = data.drop(labels='fecha',axis =1);
    x_val =data.values;
    print(columns);
    #x_val= x_vals[:,1:columns];
    y_vals = converToArray(build,contaminant);
    # Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_val= min_max_scaler.fit_transform(x_val)
    y_vals = min_max_scaler.fit_transform(y_vals)
    return [x_val, y_vals,columns];


def prepro2(data,build,contaminant):
    """
    Function to normalize the data of the neuronal network

    :param data: training data
    :param build: training data
    :param contaminant: name of the contaminant
    :return : normalize data
    """
    x_vals = data.values;
    x = x_vals.shape;
    columns = x[1];
    x_val= x_vals[:,1:columns];
    y_vals = converToArray(build,contaminant);
    return [x_val, y_vals,columns];
