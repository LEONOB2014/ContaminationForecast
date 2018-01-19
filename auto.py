from Utilites.Utilites import prepro as an
from NNSystem.neuralNetworkGpu import train as nng
import pandas as df


# est =['AJM','MGH','CCA','SFE','UAX','CUA','NEZ','CAM','LPR','SJA','CHO','IZT','SAG','TAH','ATI','FAC','UIZ','MER','PED','TLA','BJU','XAL'];
# est =['AJM','MGH','CCA','SFE','UAX','CUA','NEZ','CAM','LPR','SJA','IZT','SAG','TAH','ATI','FAC','UIZ','MER','PED','TLA','XAL'];


def trainNeuralNetworks(est, dirr, dirTrain, fechaFinal, contaminant):
    """
    Function to train the neuralNetwork of the 23 stations,
    save the training on file trainData/[nameStation].csv

    :param est: name of the station to train
    :type est: string
    :param dirr : direction of training data
    :type dirr: String
    :param dirTrain: direction of the neural network training files
    :type dirTrain: String
    :param fechaFinal: final date of training
    :type fechaFinal: date
    :param contaminant: name of the pollutants
    :type contaminant: String
    """
    tam = len(est) - 1
    tamLen = []
    i = 0
    while i <= tam:
        station = est[i]
        print(station)
        name = station + '_' + contaminant  # name the file with the data
        data = df.read_csv(dirr + name + '.csv')   # we load the data in the Variable data
        build = df.read_csv(dirr + name + '_pred.csv')  # we load the data in the Variable build
        data = data[data['fecha'] < fechaFinal]
        build = build[build['fecha'] < fechaFinal]
        tamLen.append(len(data.index))
        data = data.fillna(value=-1)
        build = build.fillna(value=-1)
        xy_values = an(data, build, contaminant)  # preprocessing
        nng(xy_values[0], xy_values[1], xy_values[2], 1000, station, contaminant, dirTrain)  # The neural network is trained
        i += 1
