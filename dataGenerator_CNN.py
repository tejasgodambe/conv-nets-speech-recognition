#!/usr/bin/python3

## Written by: D S Pavan Kumar
## Modified by: Tejas Godambe
## Date: May 2017

from subprocess import Popen, PIPE, DEVNULL
import tempfile
import pickle
import struct
import numpy
import os

## Data generator class for Kaldi
class dataGenerator_CNN:
    def __init__ (self, data, ali, exp, batchSize=256):
        self.data = data
        self.ali = ali
        self.exp = exp
        self.batchSize = batchSize

        self.maxSplitDataSize = 100
        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
 
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)
       
        self.inputFeatDim = 39 ## IMPORTANT: HARDCODED. Change if necessary.
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        
        #self.x = numpy.empty ((0, self.inputFeatDim))
        self.x = numpy.empty ((0, 1, 39, 11))
        self.y = numpy.empty ((0, self.outputFeatDim))

        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)

        ## Split data dir
        if not os.path.isdir (data + 'split' + str(self.numSplit)):
            Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()
        
        ## Save split labels and delete labels
        self.splitSaveLabels (labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
        
    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['am-info', '%s/final.mdl' % self.exp], stdout=PIPE)
        modelInfo = p1.stdout.read().splitlines()
        for line in modelInfo:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

    ## Read utterance
    def readUtterance (self, ark):
        ## Read utterance ID
        uttId = b''
        c = ark.read(1)
        if not c:
            return None, None
        while c != b' ':
            uttId += c
            c = ark.read(1)
        ## Read feature matrix
        header = struct.unpack('<xcccc', ark.read(5))
        m, rows = struct.unpack('<bi', ark.read(5))
        n, cols = struct.unpack('<bi', ark.read(5))
        featMat = numpy.frombuffer(ark.read(rows * cols * 4), dtype=numpy.float32)
        return uttId.decode(), featMat.reshape((rows,cols))
    
    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1
            labels[line[0]] = [int(i) for i in line[1:]]
        return labels, numFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + '/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                pickle.dump (splitLabels, f)

    ## Convert integer labels to binary
    def getBinaryLabels (self, intLabelList):
        numLabels = len(intLabelList)
        binaryLabels = numpy.zeros ((numLabels, self.outputFeatDim))
        binaryLabels [range(numLabels),intLabelList] = 1
        return binaryLabels
  
    ## Return a minibatch to work on
    def getNextSplitData (self):
        p1 = Popen (['splice-feats','--print-args=false','--left-context=5','--right-context=5',
                'scp:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'], stdout=PIPE)

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
    
        featList = []
        labelList = []
        while True:
            uid, featMat = self.readUtterance (p1.stdout)
            if uid == None:
                return (numpy.vstack(featList), numpy.vstack(labelList))
            if uid in labels:
                labelMat = self.getBinaryLabels(labels[uid])
                labelList.append (labelMat)
                featMat = numpy.reshape(featMat, (featMat.shape[0], 1, 39, 11))
                featList.append (featMat)

    def __iter__ (self):
        return self

    def __next__ (self):
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                break

            self.splitDataCounter += 1
            x,y = self.getNextSplitData()
            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False
        
        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize
        return (xMini, yMini)
