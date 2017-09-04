import numpy as np


def mape(real,obs):
    real = np.array(real);
    obs = np.array(obs);
    return np.mean(np.abs((real -  obs))/ real) / len(real);

def uTheils(real, obs):
    n = len(real)
    sqError = np.square(obs -real).mean();
    error = np.sqrt((1/n) * sqError);
    obsError = np.sqrt((1/n) * np.square(obs).mean());
    realError = np.sqrt((1/n) * np.square(real).mean());
    return error / (obsError + realError);

def correla(real,obs):
    return np.corrcoef(real,obs);

def metricas(real,obs,station):
    if len(real) == 0:
        met.append(station)
        met.append(0);
        met.append(0);
        met.append(0);
    else:
        met = [];
        met.append(station);
        met.append(mape(real,obs));
        met.append(uTheils(real,obs));
        met.append(correla(real,obs));
    return met;
