#!/usr/bin/env python

#import os
import sys
import argparse
import numpy as np
import caffe
import lmdb

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
def get_train_lmdb(name):
    data = np.zeros((50000,3,32,32), dtype=np.float32)
    label = np.zeros((50000,1,1,1), dtype=np.float32)
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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        help="Model definition file."
    )
    parser.add_argument(
        "--weights",
        help="Trained model weights file."
    )
    def str2bool(v):
        return v.lower() in ("yes", "True", "true", "y", "T", "t", "1")

    parser.register('type','bool',str2bool)
    parser.add_argument(
        "--multiview",
        type=int,
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--train",
        type='bool',
        help="test or train data?"
    )
    args = parser.parse_args()

    net_file = args.net
    weights_file = args.weights
    caffe.set_mode_gpu()
    net = caffe.Net(net_file, weights_file, caffe.TEST)

    if args.train:
        num = 50000
        data, label = get_train_lmdb("/home/wangzhy/data/cifar10/cifar10_train_lmdb/")
    else:
        num = 10000
        data, label = get_test_lmdb("/home/wangzhy/data/cifar10/cifar10_test_lmdb/")

    mean = get_mean("/home/wangzhy/data/cifar10/mean.binaryproto")

    inputs = {}
    if (args.multiview == 0):
        inputs['data'] = data
        inputs['data'] = inputs['data'] - mean
    elif (args.multiview == 1):
        inputs['data'] = data[:,:, 4:28, 4:28]
        inputs['data'] = inputs['data'] - mean[0, :, 4:28, 4:28]
    elif (args.multiview == 9):
        inputs['data'] = np.empty((9 * num, 3, 24, 24), dtype=np.float32)
        inputs['data'][0::9] = data[:, :, 0:24, 0:24]
        inputs['data'][1::9] = data[:, :, 0:24, 4:28]
        inputs['data'][2::9] = data[:, :, 0:24, 8:32]
        inputs['data'][3::9] = data[:, :, 4:28, 0:24]
        inputs['data'][4::9] = data[:, :, 4:28, 4:28]
        inputs['data'][5::9] = data[:, :, 4:28, 8:32]
        inputs['data'][6::9] = data[:, :, 8:32, 0:24]
        inputs['data'][7::9] = data[:, :, 8:32, 4:28]
        inputs['data'][8::9] = data[:, :, 8:32, 8:32]
        inputs['data'] = inputs['data'] - mean[0, :, 4:28, 4:28]
    elif (args.multiview == 10):
        inputs['data'] = np.empty((10 * num, 3, 24, 24), dtype=np.float32)
        inputs['data'][0::10] = data[:, :, 0:24, 0:24]
        inputs['data'][1::10] = inputs['data'][0::10, :, :, ::-1]
        inputs['data'][2::10] = data[:, :, 0:24, 8:32]
        inputs['data'][3::10] = inputs['data'][2::10, :, :, ::-1]
        inputs['data'][4::10] = data[:, :, 4:28, 4:28]
        inputs['data'][5::10] = inputs['data'][4::10, :, :, ::-1]
        inputs['data'][6::10] = data[:, :, 8:32, 0:24]
        inputs['data'][7::10] = inputs['data'][6::10, :, :, ::-1]
        inputs['data'][8::10] = data[:, :, 8:32, 8:32]
        inputs['data'][9::10] = inputs['data'][8::10, :, :, ::-1]
        inputs['data'] = inputs['data'] - mean[0, :, 4:28, 4:28]

    # test
    out = net.forward_all(**inputs)
    predictions = out['prob'];

    if (args.multiview == 0):
        predict_label = predictions.argmax(1)
        acc = predict_label == label.reshape(-1)
        print "all   : err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
    elif (args.multiview == 1):
        predict_label = predictions.argmax(1)
        acc = predict_label == label.reshape(-1)
        print "all   : err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
    elif (args.multiview == 9):
        predictions = predictions.reshape((len(predictions) / 9, 9, -1))
        predict_label = predictions[:,0].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(0,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,1].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(0,4): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,2].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(0,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,3].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(4,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,4].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(4,4): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,5].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(4,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,6].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(8,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,7].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(8,4): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,8].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "(8,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predictions = predictions.mean(1)
        predict_label = predictions.argmax(1)
        acc = predict_label == label.reshape(-1)
        print "all  : err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
    elif (args.multiview == 10):
        predictions = predictions.reshape((len(predictions) / 10, 10, -1))
        predict_label = predictions[:,0].argmax(1)
        acc = predict_label == label.reshape(-1)
        print " (0,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,1].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "-(0,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,2].argmax(1)
        acc = predict_label == label.reshape(-1)
        print " (0,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,3].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "-(0,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,4].argmax(1)
        acc = predict_label == label.reshape(-1)
        print " (4,4): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,5].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "-(4,4): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,6].argmax(1)
        acc = predict_label == label.reshape(-1)
        print " (8,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,7].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "-(8,0): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,8].argmax(1)
        acc = predict_label == label.reshape(-1)
        print " (8,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predict_label = predictions[:,9].argmax(1)
        acc = predict_label == label.reshape(-1)
        print "-(8,8): err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)
        predictions = predictions.mean(1)
        predict_label = predictions.argmax(1)
        acc = predict_label == label.reshape(-1)
        print "all   : err=%d, acc=%.2f%%, err=%.2f%%"%(num - acc.sum(), 100.*acc.sum()/num, 100-100.*acc.sum()/num)

if __name__ == '__main__':
    main(sys.argv)
