import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
import numpy as np 

class BasicDataProvider:
  def __init__(self, dataset, word_embedding=None):
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print 'BasicDataProvider: reading %s' % (features_path, )
    features_struct = scipy.io.loadmat(features_path)
    self.features = features_struct['feats']

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

    if word_embedding is not None:
      self.loadWordEmbedding2(word_embedding)
      # self.trainWordEmbedding()

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = np.random.choice(images)
    sent = np.random.choice(img['sentences'])
    #img = images[0]
    #sent = img['sentences'][0]

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def sampleImageSentencePairBatch(self, split='train', batch_size=100):
    images = self.split[split]

    imgs = np.random.choice(images, batch_size, replace=False)
    image_batch = np.zeros((batch_size, self.features.shape[0]))
    sent_batch = []

    for i, img in enumerate(imgs):
      image_batch[i,:] = self.features[:, img['imgid']]
      sent = np.random.choice(img['sentences'])
      sent_batch.append(sent['tokens'])

    return image_batch, sent_batch

  def getImageSentencePairs(self, split='val', pos=0, reverse=False):
    images = self.split[split]
    image_batch = np.zeros((len(images), self.features.shape[0]))
    sent_batch = []
    for i, img in enumerate(images):
      image_batch[i,:] = self.features[:, img['imgid']]
      # sent = np.random.choice(img['sentences'])
      sent = img['sentences'][pos]
      sent_batch.append(sent['tokens'])
      if reverse:
        sent_batch[-1][::-1]

    return image_batch, sent_batch


  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])


  def loadWordEmbedding(self, dirname='data/glove.6B/'):
    embeddings_index = {}
    f = open(os.path.join(dirname, 'glove.6B.300d.txt'))
    print ('loading word embedding vectors...')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    self.embeddings_index = embeddings_index

  def loadWordEmbedding2(self, dirname='data/GoogleNews-vectors-negative300.bin.gz', binary=True):
    from gensim.models import Word2Vec
    model = Word2Vec.load_word2vec_format(dirname, binary=True)
    self.embeddings_index = model

  def iterSentencesTokens(self):
    for img in self.dataset['images']: 
      for sent in img['sentences']:
        yield self._getSentence(sent['tokens'])

  def trainWordEmbedding(self):
    from gensim.models import Word2Vec
    sentence_iter = self.iterSentencesTokens()
    sentences = Sentence(sentence_iter)
    model = Word2Vec(sentences, size=200, window=5, min_count=2, workers=4)
    self.embeddings_index = model


  def getEmbeddedData(self, split='train', shuffle=False, max_images=-1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list

    feature_indices = np.zeros(len(imglist), dtype=int)
    sentences = np.zeros((len(imglist), len(self.embeddings_index['the'])))
    for i, idx in enumerate(ix):
      img = imglist[idx]
      feature_indices[i] = img['imgid']
      # s = 0
      # for sent in img['sentences']:
      #   s += np.asarray([self.embeddings_index[w] for w in sent['tokens'] if w in self.embeddings_index]).mean(axis=0)
      # sentences[i] = s/len(img['sentences'])
      sentences[i] = np.asarray([self.embeddings_index[w] for w in img['sentences'][0]['tokens'] if w in self.embeddings_index]).mean(axis=0)

    # feature_indices = np.asarray([imglist[i]['imgid'] for i in ix])

    return self.features[:, feature_indices].T, sentences

  def getData(self, split='train', shuffle=False, max_images=-1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list

    images = np.zeros((len(imglist)*len(imglist[0]['sentences']), self.features.shape[0]))
    sentences = np.zeros((len(imglist)*len(imglist[0]['sentences']), len(self.embeddings_index['the'])))
    for i, idx in enumerate(ix):
      img = imglist[idx]
      # feature_indices[i] = img['imgid']
      for j, sent in enumerate(img['sentences']):
        images[len(img['sentences'])*i + j] = self.features[:, img['imgid']]
        sentences[len(img['sentences'])*i + j] = np.asarray([self.embeddings_index[w] for w in sent['tokens'] if w in self.embeddings_index]).mean(axis=0)

    # feature_indices = np.asarray([imglist[i]['imgid'] for i in ix])

    return images, sentences

def getDataProvider(dataset, word_embedding=None):
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['flickr8k', 'flickr30k', 'coco'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(dataset, word_embedding)

class Sentence(object):
  """docstring for Sentence"""
  def __init__(self, sentence_iter):
    self.sentence_iter = sentence_iter

  def __iter__(self):
    # i = 0
    for sent in self.sentence_iter:
      # i += 1
      # print i
      yield sent