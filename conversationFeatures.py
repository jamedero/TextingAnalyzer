#!/usr/bin/env python3

import sys
import csv
import datetime
import argparse
from statistics import mean, median
from collections import defaultdict
import matplotlib

"""
Generates a .csv of features of a text conversation. 

Author: Julie Medero
"""

class Utterance():
    def __init__(self, speaker="", dt=None, body=""):
        self.speaker=speaker
        self.dt = datetime.datetime.strptime(dt, "%m/%d/%Y %H:%M")
        self.body = body


def main(args): 
    writer = csv.writer(args.output_file)


    headers = ("File",
               "Speaker",
               "Mean Utterance Length",
               "Median Utterance Length",
               "Mean Consecutive Utterances",
               "Median Consecutive Utterances",
               "Mean Consecutive Words",
               "Median Consecutive Words",
               "Mean Response Time",
               "Median Response Time")
    writer.writerow(headers)

    
    for input_file in args.input_files: 
        reader = csv.reader(input_file, delimiter='\t', quotechar='"')

        utterances = [Utterance(speaker, dt, body) for speaker, dt, body in reader]

        utterance_lengths = defaultdict(list)
        consecutive_texts = defaultdict(list)
        consecutive_words = defaultdict(list)
        response_time = defaultdict(list)

        last_speaker = None
        last_dt = None
        last_sequence_start = None
        num_utterances = 0
        consecutive_body_length = 0

        for utterance in utterances:
            this_speaker = utterance.speaker
            this_dt = utterance.dt
            this_body = utterance.body

            this_body_length = len(this_body.split())            
            utterance_lengths[this_speaker].append(this_body_length)

            num_utterances += 1
            
            if last_speaker is None: last_speaker = this_speaker
            if last_sequence_start is None: last_sequence_start = this_dt

            if last_speaker != this_speaker:
                consecutive_texts[last_speaker].append(num_utterances)
                consecutive_words[last_speaker].append(consecutive_body_length)
                response_time[this_speaker].append(this_dt - last_sequence_start)
                
                num_utterances = 0
                consecutive_body_length = 0
                last_sequence_start = this_dt
                
            consecutive_body_length += this_body_length
            last_speaker = this_speaker
            last_dt = this_dt

        ## Add stats for the last speaker
        consecutive_texts[last_speaker].append(num_utterances)
        consecutive_words[last_speaker].append(consecutive_body_length)
        response_time[last_speaker].append(this_dt - last_sequence_start)
            
        for speaker in utterance_lengths:
            avg_len = mean(utterance_lengths[speaker])
            med_len = median(utterance_lengths[speaker])
            
            avg_consec_utterances = mean(consecutive_texts[speaker])
            med_consec_utterances = median(consecutive_texts[speaker])

            avg_consec_words = mean(consecutive_words[speaker])
            med_consec_words = median(consecutive_words[speaker])

            response_time[speaker].sort()
            total_response_time = sum(response_time[speaker], datetime.timedelta())
            num_responses = len(response_time[speaker])
            median_response_time = response_time[speaker][int(num_responses/2)]
            
            avg_response_time = total_response_time / len(response_time[speaker])
                           

            row = (input_file.name, speaker, avg_len, med_len, avg_consec_utterances, med_consec_utterances,
                       avg_consec_words, med_consec_words, avg_response_time, median_response_time)
            writer.writerow(row)

def csv_readable(string):
     return open(string, 'r', newline='')

def csv_writeable(string):
     return open(string, 'w', newline='')
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', metavar='FILE', nargs='+',
                    help='csv file(s) to analyze', type=csv_readable)
    parser.add_argument('-o', '--output', dest='output_file', help='csv file to save analysis to', type=csv_writeable, default=sys.stdout)

    args = parser.parse_args()
    main(args)

