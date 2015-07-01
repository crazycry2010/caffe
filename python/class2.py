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
        "--model",
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
        type='bool',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--train",
        type='bool',
        help="test or train data?"
    )
    parser.add_argument(
        "--crop",
        type='bool',
        help="crop data?"
    )
    args = parser.parse_args()

    model_file = args.model
    pretrained_file = args.weights
    caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained_file, caffe.TEST)
    if args.train:
        num = 50000
        data, label = get_train_lmdb("/home/wangzhy/data/cifar10/cifar10_train_lmdb/")
    else:
        num = 10000
        data, label = get_test_lmdb("/home/wangzhy/data/cifar10/cifar10_test_lmdb/")
    mean = get_mean("/home/wangzhy/data/cifar10/mean.binaryproto")
    if args.crop:
        crop_dims = np.array([24,24])
        mean = mean[0,:,4:28,4:28]
    else:
        crop_dims = np.array([32,32])

    inputs = {}
    if args.multiview:
        start_positions = [(0,0), (0, 4), (0, 8),
                           (4, 0), (4, 4), (4, 8),
                           (8, 0), (8, 4), (8, 8)]
        end_positions = [(sy+24, sx+24) for (sy,sx) in start_positions]
        ix = 0
        inputs['data'] = np.empty((9 * len(data), 3, 24, 24), dtype=np.float32)
        for im in data:
            for i in xrange(9):
                inputs['data'][ix] = im[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
                ix += 1
    else:
        center = np.array(data[0,0,:,:].shape) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([-crop_dims / 2.0,crop_dims / 2.0])
        inputs['data'] = data[:,:, crop[0]:crop[2], crop[1]:crop[3]]
    inputs['data'] = inputs['data'] - mean
    # test
    out = net.forward_all(**inputs)
    predictions = out['prob'];
    if args.multiview:
        predictions = predictions.reshape((len(predictions) / 9, 9, -1))
        predictions = predictions.mean(1)
    predict_label = predictions.argmax(1)
    acc = predict_label == label.reshape(-1)
    print "err: %d"%(num - acc.sum())
    print "acc: %f err: %f" % (1.0*acc.sum()/num, 1-1.0*acc.sum()/num)

if __name__ == '__main__':
    main(sys.argv)
