#!/usr/bin/env python3

import csv
import sys
import re
import os
import argparse
from Texting import Dictionary, Conversation, Utterance, Normalizer
from FeatureExtractors import *

"""
Load a conversation and one or more Annotators, then process each line
in the conversation. Command-line options determind which features
are output for the conversation. 
"""

def main(args): 
    annotators = []

    thisNorm = Normalizer(args.norm)


    if args.survey:
        annotators.append(CSVFeatures(args.survey))
    if args.activedays or args.allfeatures:
        annotators.append(ActiveDays())
    if args.countwords or args.allfeatures: 
        annotators.append(CountWords())
    if args.countpos or args.allfeatures: 
        annotators.append(CountPOS())
    if args.dict: 
        thisDict = Dictionary(args.dict)
        annotators.append(DictionaryFeatureExtractor(thisDict))
    if args.responsetimes or args.allfeatures: 
        annotators.append(ElapsedTime())
    if args.timeofday or args.allfeatures: 
        annotators.append(TimeOfDay())

    need_header = True
    for csvFile in args.textfiles: 
        processFile(csvFile, args.time, thisNorm, annotators, need_header)
        need_header = False

def processFile(fname, numDays, thisNorm, annotators, need_header):

    conversation = Conversation(fname, norm=thisNorm)
    conversation.limitTimeRange(numDays)

    if annotators and conversation.utterances:
        for annotator in annotators:
            conversation.addAnnotator(annotator)

        conversation.groupFeatures()
            
        conversation.writeFeatures(args.out, need_header, os.path.basename(fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from text data files.')
    parser.add_argument('textfiles', metavar='FILE.csv', nargs='+',
                    help='a text thread in CSV file that should be analyzed')
    parser.add_argument('--dict', '-d', metavar='FILE.dic', help='a dictionary file in .dic format')
    parser.add_argument('--survey', '-s', metavar='SURVEY.csv', help='survey results file in .csv format')
    parser.add_argument('--out', '-o', metavar='FILE', help='Write results to FILE in .csv format', default="all_features.csv")
    parser.add_argument('--norm', '-n', metavar='NORM.dic', help='a tab-delimited set of replacements')
    parser.add_argument('--time', '-t', metavar='N', type=int, help='number of days to process', default=14)
    parser.add_argument('--countwords', '-w', action='store_true')
    parser.add_argument('--countpos', '-p', action='store_true')
    parser.add_argument('--responsetimes', '-r', action='store_true')
    parser.add_argument('--timehist', action='store_true')
    parser.add_argument('--timeofday', action='store_true')
    parser.add_argument('--activedays', action='store_true')
    parser.add_argument('--allfeatures', action='store_true')
    args = parser.parse_args()

    main(args)


