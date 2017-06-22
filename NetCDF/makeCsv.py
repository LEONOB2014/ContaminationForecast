from datetime import datetime, timedelta
import pandas as df
from netCDF4 import Dataset
import netCDF4 as nc4
import NewBBOX as ne
from os import listdir
import numpy as np
from pandas import concat
import re


def conver1D(array):
    """
    Function to convert an array to a list
    :param array: array with the data
    :type array = matrix float32
    :return : list with the data
    :return type: list float32
    """
    array1D = [];
    total = [];
    i = 0
    for i in range(24):
        tempData = array[i]
        for x in tempData:
            for s in x:
                array1D.append(s);
        total.append(array1D)
    return total;

def makeDates(date):
    """
    Function to create a list with the format year-month-day hours:minutes:seconds
    from 00 hours to 23 hours
    :param date : initial date
    :param type : string
    :return : list with the dates
    :return type : list datatime
    """
    listDates = [];
    date = date + ' 00:00:00';
    d =datetime.strptime(date,'%Y-%m-%d %H:%M:%S');
    listDates.append(d);
    for x in range(23):
        d = d + timedelta(hours=1);
        listDates.append(d);
    allData = df.DataFrame(listDates,columns=['fecha']);
    return allData;


def nameColumns(name,numbColumns):
    """
    Function to create list with the name of the columns
    from the variables
    :param name : Variable name
    :param type : string
    :param numbColumns : Number of columns
    :param type: int
    :return : list with the name of the columns
    :return type: list string
    """
    namesColumns=[];
    for i in range(numbColumns):
        nColumn = name+'_'+str(i);
        namesColumns.append(nColumn);
    return namesColumns;


def makeCsv(net,date):
    """
    Function to create .csv files of some variables that are in a NetCDF file,
    the .cvs file is saved in the data/NetCDF path of the project
    :param net : NetCDF file information
    :param type: NetCDF type
    :param date: initial date
    :param type: string
    """
    allData = makeDates(date);
    variables=['Uat10','Vat10','PREC2'];

    LON = net.variables['Longitude'][:];
    LAT = net.variables['Latitude'][:];

    LONsize = len(LON);
    LATsize = len(LAT);

    minlat=19.8 # 19.4284700;
    maxlat=-19.033333#20;
    minlon=-99.933333#-99.127660;
    maxlon=99.366667#-98;

    celda = [];
    var_cut=[];
    for i in variables:
        var= net.variables[i][:]
        celda.append(var);
        result = ne.NewBBOX(var,LON,LAT,LONsize,LATsize,minlat,maxlat,minlon,maxlon);
        var_cut.append(result[0]);


    for ls in range(len(var_cut)):
        temp = conver1D(var_cut[ls]);
        dataMatrix= np.array(temp);
        name = variables[ls]+'_'+date+'.csv'
        myIndex = nameColumns(variables[ls],len(temp[0]));
        tempFrame =df.DataFrame(dataMatrix,columns=myIndex);
        allData = concat([allData,tempFrame], axis=1);
        allData.to_csv('../data/NetCDF/'+name,encoding='utf-8',index= False);


def readFiles():
    """
    Function to read all NetCDF files that are in the specified path
    and named by the format Dom1_year-month-day.nc
    """
    dirr = '/home/pablo/DATA/' #specified path
    date = '\d\d\d\d-\d\d-\d\d'
    name = 'Dom1_'
    patron = re.compile(name+'.*')
    patron2 = re.compile(date);
    for x in listdir(dirr):
        if patron.match(x) != None:
            ls= dirr +x;
            print(ls);
            f = patron2.findall(x);
            net = Dataset(ls);
            makeCsv(net,f[0]);

readFiles();
#makeDates('2017-06-13');
