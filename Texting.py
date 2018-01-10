import sys
import re
from collections import defaultdict
from emoticons import analyze_tweet as emoticons
import unicodedata
from twokenize_wrapper import tokenize
import datetime
import csv
import numpy as np
import os.path


"""
Collection of convenience classes for processing text message conversations. 

Utterance: A single line of a texting/messaging conversation (a text). 
Conversation: A set of Utterances between two people. 
Dictionary: An object that provides the interface for e.g. LIWC dictionaries 
    that map every word to zero or more categories. 
Normalizer: Data cleansing helper for dealing with informal text. 
"""

class Normalizer():
    def __init__(self, fname=None):
        self.pairs = {}

        if fname: 
            self.loadFile(fname)
    def loadFile(self, fname): 
        for line in open(fname): 
            try: 
                key, replacement = line.strip().split("\t")
                key = re.compile(key)
            except: 
                sys.exit("problem with {}".format(line))
            self.pairs[key] = replacement
    def replace(self, word): 
        for ch in word: 
            if unicodedata.category(ch) == "Emoticon": 
                word = word.replace(ch, " {} ".format(EMOTICONAAA))

        for stem in self.pairs: 
            if stem.fullmatch(word):
                return self.pairs[stem]
        return word

class Utterance():
    def __init__(self, speaker="", dt=None, body="", norm=None):
        try: 
            self.normalizer=norm
            self.speaker=speaker.strip()
            self.dt = datetime.datetime.strptime(dt.strip(), "%m/%d/%Y %H:%M")
            self.body = body.strip()
        except: 
            sys.exit("Couldn't parse utterance speaker={}, dt={}, body={}".format(speaker, dt, body))

        self.cleanTokens()

        self.features = []

    def addFeatures(self, features): 
        self.features += features

    def uniqueTokens(self): 
        return Counter(self.lower_tokens)

    def cleanHelper(self, body): 
        tokens = tokenize(body)
        tokens = [x.lower().strip() for x in tokens]
        tokens = [x for x in tokens if emoticons(x)=="NA"]
        tokens = [x.strip(" #\-*!._(){}~,^") for x in tokens]
        tokens = [self.normalizer.replace(x) for x in tokens]
        tokens = [x for x in tokens if re.search("\w", x)]
        
        body = " ".join(tokens)
        return tokens, body

    def cleanTokens(self): 
        for quotechar in ["â", "’", ""]:
            self.body = self.body.replace(quotechar, "'")
        for encodingissue in ["ð'",]: 
            self.body = self.body.replace(encodingissue,"")
        for spacechar in ["/","-", "'d"]: 
            self.body = self.body.replace(spacechar, " " + spacechar + " ")
        
        self.lower_tokens, self.body = self.cleanHelper(self.body)
        self.lower_tokens, self.body = self.cleanHelper(self.body)
        self.lower_tokens, self.body = self.cleanHelper(self.body)

class Conversation():
    def __init__(self, fname=None, norm=None): 
        self.utterances = []
        self.annotators = []
        self.normalizer = norm
        self.fname = fname
        if fname is not None: 
            self.loadFile(fname)

    def participantName(self): 
        return os.path.splitext(os.path.basename(self.fname))[0]

    def loadFile(self, fname): 
        reader = csv.reader(open(fname, newline=''), delimiter="\t", quotechar='"')

        for i, row in enumerate(reader): 
            name, dt, body = row
            self.addUtterance(name, dt, body)
        self.utterances = [x for x in self.utterances if x.lower_tokens]

    def addAnnotator(self, annotator): 
        self.annotators.append(annotator)
        annotator.doFeatures(self)

    def addUtterance(self, name, dt, body): 
        self.utterances.append(Utterance(name, dt, body, norm=self.normalizer))

    def lastUtterance(self): 
        return self.utterances[-1].dt
    def limitTimeRange(self, numDays): 
        if numDays > 0:
            max_threshold = datetime.datetime.combine(self.lastUtterance().date(), datetime.time())
            min_threshold = max_threshold - datetime.timedelta(days=numDays)

            self.utterances = [x for x in self.utterances if x.dt > min_threshold and x.dt < max_threshold]
    def uniqueTokens(self): 
        tokens = Counter()
        for utterance in self.utterances: 
            tokens.update(utterance.uniqueTokens())
        return tokens

    def groupFeatures(self):
        self.features = None
        self.heading = ["Conversation", ]

        for annotator in self.annotators: 
            this_heading, this_features = annotator.groupFeatures(self.utterances)

            self.heading += this_heading

            if self.features is None: 
                self.features = this_features
            else: 
                self.features = np.hstack([self.features, this_features])

    def writeFeatures(self, fname, need_header, conversation_name):
        writer = csv.writer(open(fname, newline='', mode='a'), delimiter=",", quotechar='"')

        if need_header:
            writer.writerow(self.heading)

        writer.writerow([os.path.splitext(conversation_name)[0],] + list(self.features))

    def writeTimeHist(self, fname):
        writer = csv.writer(open(fname, newline='', mode='a'), delimiter="\t", quotechar='"')
        writer.writerow(["Conversation", "Speaker",] + ["Words {}".format(i) for i in range(24)] + ["Messages {}".format(i) for i in range(24)])
                            
        people = set([x.speaker for x in self.utterances])
        for person in people:
            per_utters = [(u.dt.time().hour, len(u.lower_tokens)) for u in self.utterances if u.speaker == person]
            word_counts = [sum([words for hour, words in per_utters if hour == i]) for i in range(24)]
            text_counts = [len([hour for hour, words in per_utters if hour == i]) for i in range(24)]
            writer.writerow([self.fname, person,] + word_counts + text_counts)

class Dictionary(): 
    def __init__(self, fname): 
        self.categories = {}
        self.words = defaultdict(list)
        self.parseFile(fname)

    def parseFile(self, fname): 
        self.readingIndeces = False
        self.readingVocab = False

        for line in open(fname): 
            self.processDictLine(line.strip())

    def processDictLine(self, line): 
        if line == '%': 
            if not(self.readingIndeces): 
                self.readingIndeces = True
            else: 
                self.readingVocab = True
                self.readingIndeces = False
        elif self.readingIndeces: 
            key, label = line.split("\t")
            self.categories[key] = label
        else: 
            fields = line.split("\t")
            stem = fields[0].replace("*", ".*")
            categories = fields[1:]
            self.words[re.compile(stem)] = categories

    def categoryNames(self): 
        return sorted(list(self.categories.values()))

    def hasMatch(self, word): 
        for stem in self.words: 
            if stem.fullmatch(word): return True
        return False

    def vocabulary(self): 
        return set([re.compile(x) for x in self.words.keys()])

    def countsByCategory(self, tokens): 
        categories = sum([self.findMatchingCategories(word) for word in tokens],[])
        return dict((x, categories.count(x)) for x in self.categoryNames())

    def findMatchingCategories(self, word): 
        word = word.strip().lower()
        word = re.sub("\W", "", word)

        retval = []
        for stem, categories in self.words.items(): 
            if stem.fullmatch(word): 
                retval += categories
        return [self.categories[key] for key in retval]
