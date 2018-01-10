from CMUTweetTagger import runtagger_parse
import re
from collections import defaultdict
import numpy as np
import csv

"""
Each FeatureExtractor calculates a property of each line (Utterance) of 
a text Conversation, then aggregates that property across the entire
conversation. 

Author: Julie Medero
"""

class FeatureExtractor():
    def __init__(self):
        self.heading = []
        self.groupby = "Day"
        self.features = []
        self.normalize = False
    def header(self):
        return self.heading
    def doFeatures(self, conversation):
        self.features = [self.doUtteranceFeatures(x) for x in conversation.utterances]

    def doUtteranceFeatures(self, utterance): 
        return []

    def groupFeatures(self, utterances): 
        if self.groupby == "Day": 
            return self.groupFeaturesByDay(utterances) 
        elif self.groupby == "Conversation": 
            return (self.heading, self.features) 
        else: 
            sys.exit("Unknown groupby: {}".format(self.groupby))

    def groupFeaturesByDay(self, utterances):        
        people = set([x.speaker for x in utterances])
        features = None
        headings = []

        for person in sorted(people):
            person_features = {}
            for utterance, u_features in zip(utterances, self.features):
                if utterance.speaker != person: continue
                day = utterance.dt.date()
                this_features = np.array(u_features)
                if day not in person_features:
                    person_features[day] = u_features
                else:
                    person_features[day] += np.array(u_features)
                    
            if self.normalize: 
                for day, d_features in person_features.items():
                    d_features[1:] = d_features[1:] / d_features[0]

            this_features = np.vstack(list(person_features.values()))
            means = this_features.mean(axis=0)
            variances = this_features.var(axis=0)

            if features is None: 
                features = np.hstack([means, variances])
            else: 
                features = np.hstack([features, means, variances])

            headings += ["{}_{}_Mean".format(person, feature_name) for feature_name in self.header()]
            headings += ["{}_{}_Variance".format(person, feature_name) for feature_name in self.header()]
            
        return (headings, features)



class DictionaryFeatureExtractor(FeatureExtractor): 
    def __init__(self, dict): 
        super().__init__()
        self.dictionary = dict
        self.groupby = "Day"
        self.normalize = False
    def header(self): 
        return sorted(self.dictionary.categoryNames())
    def doUtteranceFeatures(self, utterance): 
        counts = self.dictionary.countsByCategory(utterance.lower_tokens)
        return [counts[x] for x in self.header()]

class CountWords(FeatureExtractor): 
    def __init__(self): 
        super().__init__()
        self.groupby = "Day"
        self.normalize = False
        self.heading = ["All_Words", "Short_Words"]
    def doUtteranceFeatures(self, utterance):
        tokens = utterance.lower_tokens
        num_tokens = len(tokens)

        # words < 6 letters
        short_tokens = [x for x in tokens if len(x) < 6]

        num_short = len(short_tokens)

        return [num_tokens, num_short]

    
class CountPOS(FeatureExtractor): 
    def __init__(self): 
        super().__init__()
        self.groupby = "Day"
        self.normalize = False
        self.heading = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 
                        'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 
                        'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'URL', 'VB', 
                        'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

        self.tagger_cmd = "java -XX:ParallelGCThreads=2 -Xmx500m -jar lib/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar"

    def doFeatures(self, conversation): 
        tweets = [" ".join(u.lower_tokens) for u in conversation.utterances]
        parse = runtagger_parse(tweets, self.tagger_cmd)
        self.features = [self.doCounts(x) for x in parse]

    def doCounts(self, listOfTuples): 
        tags = [tag for word, tag, likelihood in listOfTuples]
        missing = set([tag for word, tag, linelihood in listOfTuples if tag not in self.heading])
        return [tags.count(x) for x in self.header()]

class ElapsedTime(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.groupby = "Conversation"
        self.normalize = False

    def doFeatures(self, conversation):
        elapsedTimes = defaultdict(list)

        prev_utterance = conversation.utterances[0]

        for utterance in conversation.utterances[1:]:
            this_speaker = utterance.speaker
            time_elapsed = utterance.dt - prev_utterance.dt
            if utterance.dt < prev_utterance.dt: 
                print(utterance.dt, prev_utterance.dt)
            elapsedTimes["{}-{}".format(prev_utterance.speaker, utterance.speaker)].append(time_elapsed.total_seconds()/60)
            prev_utterance = utterance

        self.heading = []
        all_features = []

        for heading, features in sorted(elapsedTimes.items()):
            # count
            self.heading.append("{}_Count".format(heading))
            all_features.append(len(features))

            features = np.array(features)

            # min
            self.heading.append("{}_Min".format(heading))
            all_features.append(features.min())
            
            # max
            self.heading.append("{}_Max".format(heading))
            all_features.append(features.max())
            
            # mean
            self.heading.append("{}_Mean".format(heading))
            all_features.append(features.mean())

            # variance
            self.heading.append("{}_Variance".format(heading))
            all_features.append(features.var())
        

        self.features = all_features
    
class CountEmoji(FeatureExtractor): 
    def __init__(self): 
        self.heading = []
    def doUtteranceFeatures(self, utterance): 
        return []

class ActiveDays(FeatureExtractor): 
    def __init__(self): 
        super().__init__()
        self.groupby = "Conversation"
        self.normalize = False

    def doFeatures(self, conversation): 
        headings = []
        features = []
                            
        people = set([x.speaker for x in conversation.utterances])
        for person in people:
            this_days = set([u.dt.date() for u in conversation.utterances if u.speaker == person])
            headings.append("{}_Days_Active".format(person))
            features.append(len(this_days))

        self.heading = headings
        self.features = np.array(features)

class CSVFeatures(FeatureExtractor): 
    def __init__(self, fname): 
        super().__init__()
        self.groupby = "Conversation"
        self.normalize = False

        self.content = {}
        reader = csv.reader(open(fname, newline=''), delimiter=",", quotechar='"')

        self.heading = []

        for row in reader: 
            if not(self.heading): 
                self.heading = row[1:]
            else: 
                id = row[0]
                self.content[id] = row[1:]

    def doFeatures(self, conversation): 
        heading = []
        features = []

        participant_id = conversation.participantName()
        if participant_id in self.content:
            self.features = self.content[participant_id]
        else:
            self.features = [-1,] * len(self.heading)

class TimeOfDay(FeatureExtractor): 
    def __init__(self): 
        super().__init__()
        self.groupby = "Conversation"
        self.normalize = False

    def doFeatures(self, conversation): 

        heading = []
        features = []
                            
        people = set([x.speaker for x in conversation.utterances])
        for person in people:
            per_utters = [(u.dt.time().hour, len(u.lower_tokens)) for u in conversation.utterances if u.speaker == person]
            word_counts = [sum([words for hour, words in per_utters if hour == i]) for i in range(24)]
            text_counts = [len([hour for hour, words in per_utters if hour == i]) for i in range(24)]

            heading += ["{}_Words_Hour_{}".format(person, i) for i in range(24)]
            heading += ["{}_Messages_Hour_{}".format(person, i) for i in range(24)]

            features += word_counts + text_counts

        self.heading = heading
        self.features = np.array(features)
