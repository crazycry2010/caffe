#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import caffe
import lmdb
import time

def get_test_lmdb(name):
    data = np.zeros((10000,3,32,32), dtype=np.float32)
    label = np.zeros((10000,1,1,1), dtype=np.float32)
    db = lmdb.open(name)
    txn= db.begin()
    cursor = txn.cursor()
    for idx, value in enumerate(cursor.iternext()):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value[1])
        array = caffe.io.datum_to_array(datum)
        label[idx] = datum.label
        data[idx] = array
    return data, label
def get_mean(name):
    file=open(name)
    data=file.read()
    blobproto=caffe.proto.caffe_pb2.BlobProto()
    blobproto.ParseFromString(data);
    mean=caffe.io.blobproto_to_array(blobproto)
    return mean
def save_test():
    data, label = get_test_lmdb("/home/wangzhy/data/cifar10/cifar10_test_lmdb/")
    np.save("/home/wangzhy/data/cifar10/cifar10_test_data.npy",data);
    np.save("/home/wangzhy/data/cifar10/cifar10_test_label.npy",label);
def save_mean():
    get_mean("/home/wangzhy/data/cifar10/mean.binaryproto")
    np.save("/home/wangzhy/data/cifar10/mean.npy",mean.reshape(mean.shape[1:]))   

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )
    def str2bool(v):
        return v.lower() in ("yes", "True", "true", "y", "T", "t", "1")
    parser.register('type','bool',str2bool)
    parser.add_argument(
        "--center_only",
        type='bool',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    args = parser.parse_args()
    
    caffe_root = '/home/wangzhy/cuda-workspace/caffe_release/'
    #model_file = caffe_root + 'examples/cifar10/86_local/deploy.prototxt'
    #pretrained_file = caffe_root + 'examples/cifar10/86_local/cifar10_11_iter_240000.caffemodel'
    model_file = args.model_def
    pretrained_file = args.pretrained_model
    caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained_file, caffe.TEST)
    #print net.inputs
    #print net.outputs 
    data, label = get_test_lmdb("/home/wangzhy/data/cifar10/cifar10_test_lmdb/")
    mean = get_mean("/home/wangzhy/data/cifar10/mean.binaryproto")    
    crop_dims = np.array([24,24])
    inputs = {}
    oversample = not args.center_only
    if oversample:
        inputs['data'] = caffe.io.oversample(data.transpose(0,2,3,1), crop_dims)
        inputs['data'] = inputs['data'].transpose(0,3,1,2)
        inputs['label'] = np.tile(label,(1,10)).reshape(-1,1,1,1)
    else:
        center = np.array(data[0,0,:,:].shape) / 2.0 
        crop = np.tile(center, (1, 2))[0] + np.concatenate([-crop_dims / 2.0,crop_dims / 2.0])
        inputs['data'] = data[:,:, crop[0]:crop[2], crop[1]:crop[3]]
        inputs['label'] = label       
    inputs['data'] = inputs['data'] - mean[0,:,4:28,4:28]
    # test
    out = net.forward_all(**inputs)
    predictions = out['prob'];
    if oversample:
        predictions = predictions.reshape((len(predictions) / 10, 10, -1))
        predictions = predictions.mean(1)
    predict_label = predictions.argmax(1)
    acc = predict_label == label.reshape(-1)
    print acc.sum()/10000.0
    
if __name__ == '__main__':
    main(sys.argv)