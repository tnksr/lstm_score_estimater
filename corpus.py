# coding: utf-8
import numpy as np
from random import shuffle
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

class Corpus(object):
    def __init__(self, min_length=0, tokenizer=' ', preprocessor=None):
        self.size = 0
        self.min_length = min_length
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        # 0 for padding
        self.word_to_id = {'<UNK>': 0}
        self.id_to_word = {0: '<UNK>'}
        self.star_count = {}
    
    def __call__(self, input_file, batch_size=None, laguage=False, delimiter=None, debug=False):
        source_corpus = []
        target_corpus = []
        
        if delimiter is not None:
            pass
        elif input_file.split('.')[-1] == 'tsv':
            delimiter = '\t'
        elif input_file.split('.')[-1] == 'csv':
            delimiter = ','
        
        if batch_size:
            loader = self.batch(input_file, batch_size, delimiter)
        else:
            loader = self.shuffle(input_file, delimiter)
        
        for star, text in loader:
            source_corpus.append(text)
            target_corpus.append(star)
        
        if debug:
            print('corpus  size    : %d' % self.size)
            print('vocabulary size : %d' % len(self.word_to_id))
            print('star count')
            print('    +---+-------+')
            for star, count in sorted(self.star_count.items(), key=lambda x:x[0]):
                print('    | %d |  %d|' % (star, count))
            print('    +---+-------+')
        
        return source_corpus, target_corpus
        

    def generator(self, input_file, delimiter='\t', indexs={'star': 0, 'text': 1}):
        with open(input_file) as f:
            for line in f:
                line = line.strip().split(delimiter)
                if len(line) != len(indexs): continue
                star, text = line
                
                if self.preprocessor:
                    text = self.preprocessor(text)
                else:
                    text = text.split(self.tokenizer)
                
                if len(text) < self.min_length: continue
                for word in text:
                    if word not in self.word_to_id:
                        self.word_to_id[word] = len(self.word_to_id)
                        self.id_to_word[self.word_to_id[word]] = word
                    yield int(star), self.convert(text)
    
    def shuffle(self, input_file, delimiter='\t', indexs={'star': 0, 'text': 1}):
        all_pairs = {}
        for star, text in self.generator(input_file, delimiter, indexs):
            if star not in all_pairs:
                all_pairs[star] = []
                self.star_count[star] = 0
            all_pairs[star].append(text)
        
        sample_num = min([len(texts) for texts in all_pairs.values()])
        sample_num = sample_num + sample_num // 10
        corpus = []
        for star, texts in all_pairs.items():
            choose_idx = np.random.choice(len(texts), min(len(texts), sample_num), replace=False)
            self.star_count[star] = min(len(texts), sample_num)
            corpus += [(star-1, text) for i, text in enumerate(texts) if i in choose_idx]
        shuffle(corpus)
        self.size = len(corpus)
        return corpus
    
    def batch(self, input_file, batch_size, delimiter='\t', indexs={'star': 0, 'text': 1}):
        corpus = self.shuffle(input_file, delimiter, indexs)
        corpus = sorted(corpus, key=lambda x:len(x[1]))
        batch_corpus = []
        for start in range(0, self.size - batch_size, batch_size):
            end = start+batch_size
            max_length = len(corpus[end-1][1])
            star_batch = []
            text_batch = []
            for star, text in corpus[start:end]:
                star_batch.append(star)
                text_batch.append(text + [0]*(max_length-len(text)))
            batch_corpus.append((star_batch, text_batch))
        shuffle(batch_corpus)
        for star_batch, text_batch in batch_corpus:
            yield star_batch, text_batch
                
    def convert(self, text):
        return [self.word_to_id[word] if word in self.word_to_id else 1 for word in text]
    
    def revert(self, ids):
        # TODO: EOS
        return [self.id_to_word[i] if int(i) < len(self.id_to_word) else '<UNK>' for i in ids]
    
    def to_file(self, input_file, output_file, delimiter='\t', indexes={'star': 0, 'text': 1}):
        with open(output_file, 'w') as out_f:
            for star, text in self.generator(input_file, delimiter=delimiter, indexes=indexes):
                line = ['%d' % star, self.tokenizer.join(text)]
                out_f.write(delimiter.join(line) + '\n')


if __name__ == '__main__':
    print ('corpus.py')
