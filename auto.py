from Utilites.Utilites import prepro as an
from Utilites.FormatData import FormatData as fd
from tests.neuralNetwork import train as nn
import pandas as df
import numpy as np
import tensorflow as tf

est =['AJM','ATI','BJU','CAM','CCA','CHO','CUA','FAC','IZT','LPR','MER','MGH','NEZ','PED','SAG','SFE','SJA','TAH','TLA','UAX','UIZ','XAL'];
contaminant = 'O3';


def trainNeuralNetworks():
    """
    Function to train the neuralNetwork of the 23 stations,
    save the training on file trainData/[nameStation]
    """
    i=0;
    while i <= 2:
        station = est[i];
        print(station);
        name = station +'_'+contaminant; #name the file with the data
        data = df.read_csv('data/'+name+'.csv', delim_whitespace =True); #we load the data in the Variable data
        build = df.read_csv('data/'+name+'_pred.csv',delim_whitespace = True); #we load the data in the Variable build
        xy_values = an(data,build, contaminant); # preprocessing
        nn(xy_values[0],xy_values[1],xy_values[2],1000,station,contaminant); #The neural network is trained
        i+=1;



def init_weight(shape):
    """
    Function for the define Variable function weight
    :param shape: Matrix containing weight
    :type shape : matrix float32

    :return: matrix weight
    """
    weight = tf.Variable(tf.random_normal(shape));
    return weight;

def init_bias(shape):
    """
    Function for the define Variable function weight
    :param shape: Matrix containing bias
    :type shape : matrix float32
    :return: matrix bias
    """
    bias=  tf.Variable(tf.random_normal(shape));
    return bias;

def fully_connected(input_layer,weight,biases):
    layer = tf.add(tf.matmul(input_layer,weight), biases);
    return tf.nn.sigmoid(layer);

def prediction(station, date,contaminant):
    """
    Function to obtain a prediction of a neural network that has
    already been trained previously
    :param station: station name for the prediction
    :type station: string
    :param date: date for the prediction
    :type date: string format Years/month/day
    :param contaminant: contaminant for the prediction
    :type contaminant: string
    :return : value for the prediction
    """
    name = 'train_'+station+'_'+contaminant+'';
    data = df.read_csv('data/'+station+'_'+contaminant+'.csv', delim_whitespace =True);
    x_vals = data.values;
    x = x_vals.shape;
    columns = x[1];
    x_vals= x_vals[:,1:columns];
    print(x_vals);

    x_data = tf.placeholder(shape=[None,columns-1],dtype=tf.float32);
    y_target= tf.placeholder(shape=[None,1],dtype =tf.float32);
    #--------Create the first layer (size hidden nodes)--------
    # TODO ya recibe todas las columnas en la primera capa
    weight_1 = init_weight(shape=[columns-1,columns-1]);
    bias_1 = init_bias(shape=[columns-1]);
    layer_1 = fully_connected(x_data,weight_1,bias_1);

    #--------Create the second layeprint(size);--------
    weight_2 = init_weight(shape=[columns-1,(columns-1)*2]);
    bias_2= init_bias(shape=[(columns-1)*2]);
    layer_2 = fully_connected(layer_1,weight_2, bias_2);


    #--------Create output layer (1 output value)--------
    weight_3= init_weight(shape=[(columns-1)*2,1]);
    bias_3 = init_bias(shape=[1]);
    final_output = fully_connected(layer_2,weight_3, bias_3);

    # Declare loss function (L1)
    loss= tf.reduce_mean(tf.abs(y_target - final_output));

    # Declare optimizer gradientDescent
    my_opt = tf.train.GradientDescentOptimizer(0.1);
    train_step = my_opt.minimize(loss);
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'trainData/'+station+'/'+name+'.ckpt'); #we load the training
        print(sess.run(final_output, feed_dict={x_data: np.array([[14.0,7.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0,3.0,21.0]])}));

#trainNeuralNetworks();
prediction('AJM','2016/01/08','O3');
