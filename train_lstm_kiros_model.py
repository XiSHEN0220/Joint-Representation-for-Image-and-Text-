import numpy as np
import theano 
from keras.layers import Input, Embedding, LSTM, Dense, Merge, Masking, Activation, GlobalMaxPooling1D, Convolution1D, Dropout, Reshape, TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras import backend as K
from utils import RawSentGenerator, load_embedding, DataGenerator, Merge3DTensors, Cutting
from objectives import kiros_loss
from data_provider import getDataProvider
import time
import json
import copy
from scipy.spatial.distance import cdist
import os
import argparse
import pandas as pd
from pandas import DataFrame

def text2image(y):
	idx = np.argsort(y,axis=1)
	n = idx.shape[0]
	ranks = np.zeros(n)
	for i in range(n):
		ranks[i] = find_pos([i], idx[i, :])
	return ranks

def find_pos(query_list, search_list):
	for i in range(len(search_list)):
		if search_list[i] in query_list:
			break
	return i+1

def image2text(y):
	idx = np.argsort(y, axis=1)
	n = idx.shape[0]
	ranks = np.zeros(n)
	for i in range(n):
		ranks[i] = find_pos(np.arange(i, 5*n, n), idx[i, :])
	return ranks

def rnn_predict(imgfeat, samples, model, maxlen):
	data = pad_sequences(samples, maxlen=maxlen, padding='post') 
	return model.predict([np.repeat(imgfeat, data.shape[0], axis=0), data], verbose=0)
	

def beamsearch(img_feat, model, maxlen, vocab_size, predict=rnn_predict,
			   k=20, use_unk=False, eos=0):
	"""return k samples (beams) and their NLL scores, each sample is a sequence of labels,
	all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
	You need to supply `predict` which returns the label probability of each sample.
	`use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
	"""
	
	dead_k = 0 # samples that reached eos
	dead_samples = []
	dead_scores = []
	live_k = 1 # samples that did not yet reached eos
	live_samples = [[vocab_size]]
	live_scores = [0]

	while live_k and dead_k < k:
		# for every possible live sample calc prob for every possible label 
		probs = predict(img_feat, live_samples, model, maxlen)
		# total score for every sample is sum of -log of word prb
		cand_scores = np.array(live_scores)[:,None] - np.log(probs)

		cand_flat = cand_scores.flatten()

		# find the best (lowest) scores we have from all possible samples and new words
		ranks_flat = cand_flat.argsort()[:(k-dead_k)]
		live_scores = cand_flat[ranks_flat]
		# append the new words to their appropriate live sample
		voc_size = probs.shape[1]
		live_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ranks_flat]

		# live samples that should be dead are...
		zombie = [s[-1] == eos or len(s) >= maxlen for s in live_samples]
		
		# add zombies to the dead
		dead_samples += [s for s,z in zip(live_samples,zombie) if z]  # remove first label == empty
		dead_scores += [s for s,z in zip(live_scores,zombie) if z]
		dead_k = len(dead_samples)
		# remove zombies from the living 
		live_samples = [s for s,z in zip(live_samples,zombie) if not z]
		live_scores = [s for s,z in zip(live_scores,zombie) if not z]
		live_k = len(live_samples)
	
	outsamples = dead_samples + live_samples
	outscores =  dead_scores + live_scores
	outscores = np.array([score/(len(sample) - 1) for sample, score in zip(outsamples, outscores)])
	idx = np.argsort(outscores)

	return [outsamples[i] for i in idx], outscores[idx]

def train_model(random_seed, 
				dataset, 
				batch_size, 
				nb_epochs, 
				latent_space_dim, 
				hidden_state_decoder, 
				sentence_maxlen, 
				embed_size, 
				vocab_size, 
				learning_rate, 
				drop_out1, 
				model_number, 
				weight,
				drop_out2 = False,
				optimizer = 'Adam'):
	
	reverse_sentence = True

	dp = getDataProvider(dataset)
	image_dim = dp.features.shape[0]
	np.random.seed(random_seed)
	model_number = str(model_number)

	output_json = 'tuning/'+dataset+'_model_' + model_number + '.json'
	print 'model architecture will be written to %s...' %output_json
	output_h5 = 'tuning/'+dataset+'_model_' + model_number + '.h5'
	print 'model weight will be written to %s...' %output_h5


	print('Preprocessing texts ...')
	tokenizer = Tokenizer(nb_words=vocab_size, char_level=True)
	tokenizer.fit_on_texts(RawSentGenerator(dp, ['train']))
	word_index = tokenizer.word_index
	print('word count: %d' % len(word_index))

	# add "start" to vocabulary
	vocab_size += 1

	print('Building model ...')

	## building Ranking network
	MAIN_INPUT_SENTENCE = Input(shape=(sentence_maxlen, ), dtype='int32', name='input_sentence')

	embed_layer = load_embedding(word_index, vocab_size=vocab_size, input_length=sentence_maxlen, dim=embed_size, embedding='glove', mask_zero=True)
	embedding = embed_layer(MAIN_INPUT_SENTENCE)
	
	if drop_out1 :  
		lstm_ranking = LSTM(latent_space_dim, activation='tanh', return_sequences=False, name='lstm_encoder', dropout_W = 0.5, dropout_U = 0.5)(embedding)
	else :
		lstm_ranking = LSTM(latent_space_dim, activation='tanh', return_sequences=False, name='lstm_encoder')(embedding)

	text_encode = lstm_ranking

	MAIN_INPUT_IMAGE = Input(shape=(image_dim, ), name='input_image')

	if drop_out2 : 
		image_before_dropout = Dense(latent_space_dim, name='image_out', W_regularizer=l2(1e-3))(MAIN_INPUT_IMAGE)
		#image_encode = Dropout(0.5)(img_feature)
		image_encode = image_before_dropout
	else :
		image_encode = Dense(latent_space_dim, name='image_out', W_regularizer=l2(1e-3))(MAIN_INPUT_IMAGE)

	output_encoder = Merge(mode='concat', concat_axis=1)([image_encode, text_encode])

	##building Decoder
	input_image_reshape = Reshape((1, latent_space_dim), name='reshape')(image_encode)

	merge_image_sentence = Merge3DTensors(mode='concat', concat_axis=1, name='merge')([input_image_reshape, embedding])

	if drop_out2 : 
		lstm_decoder = LSTM(hidden_state_decoder, activation='tanh', return_sequences=True, name='lstm_decoder', dropout_W = 0.5, dropout_U = 0.5)(merge_image_sentence)
	else :
		lstm_decoder = LSTM(hidden_state_decoder, activation='tanh', return_sequences=True, name='lstm_decoder')(merge_image_sentence)

	# word distribution output excluding 'start' index
	output_decoder = TimeDistributed(Dense(vocab_size, activation='softmax'), name='softmax_out')(lstm_decoder)

	output_decoder = Cutting(name='output')(output_decoder)

	model = Model(input=[MAIN_INPUT_IMAGE, MAIN_INPUT_SENTENCE], output=[output_encoder ,output_decoder])

	embed_layer.trainable = True
	with open(output_json, 'w') as outfile:
		json.dump(model.to_json(), outfile)

	if optimizer == 'rmsprop' :  
		rmsprop_optimizer = RMSprop(lr=learning_rate)	
		model.compile(optimizer=rmsprop_optimizer, loss=[kiros_loss, 'sparse_categorical_crossentropy'], loss_weights=[weight, 1])
	else :
		adam_optimizer = Adam(lr=learning_rate)
		model.compile(optimizer=adam_optimizer, loss=[kiros_loss, 'sparse_categorical_crossentropy'], loss_weights=[weight, 1])


	print('Start training ...')
	epoch = 0

	opt_val0 = np.inf
	opt_val1 = np.inf
	opt_total = np.inf
	nb_epoch_no_decreasing = 0

	n_train = dp.getSplitSize('train', ofwhat='images')
	n_batches = -(-n_train//batch_size)
	print ('n_batches: %d' % n_batches)

	# validation data
	image_val, text_val = dp.getImageSentencePairs('val', reverse=reverse_sentence)
	image_val = image_val.astype('float32')
	text_val = tokenizer.texts_to_sequences(text_val)
	text_val = pad_sequences(text_val, maxlen=sentence_maxlen, padding='post')

	output_val = text_val
	output_val = np.expand_dims(output_val, -1)

	text_val = np.hstack([vocab_size*np.ones((text_val.shape[0], 1), dtype=text_val.dtype), text_val[:, :-1]])
	
	break_while = False

	while epoch < nb_epochs:
		epoch += 1
		i = 0
		for image_train_batch, text_train_batch in DataGenerator(dp, batch_size, tokenizer, sentence_maxlen, n_batches, split='train', padding='post', shuffle=True, reverse=reverse_sentence):
			# create image and text matrix for the given batch

			output_train_batch = text_train_batch
			## add a start for a trial
			text_train_batch = np.hstack([ vocab_size *np.ones((text_train_batch.shape[0], 1), dtype=text_train_batch.dtype), text_train_batch[:, :-1]])

			output_train_batch = np.expand_dims(output_train_batch, -1)

			cost = model.train_on_batch([image_train_batch, text_train_batch], [np.zeros(image_train_batch.shape[0]), output_train_batch])

			print ('at epoch %d: %d/%d batch done, kiros loss %f, decoder loss %f' % (epoch, i+1, n_batches, cost[1], cost[2]))
			del text_train_batch
			del image_train_batch
			del output_train_batch

			# perform validation
			if epoch <= 20 :
				i += 1
				if i >= n_batches:
					break
				print 'Epoch %d'%epoch
				continue
			else :
				val_loss = model.evaluate([image_val, text_val], [np.zeros(image_val.shape[0]), output_val], batch_size=batch_size, verbose=0)
		
			print('validation total loss is %f, %f, total validation loss is %.2f' % (val_loss[1], val_loss[2], val_loss[1] * weight + val_loss[2]))
			'''if val_loss[1] < opt_val0 and val_loss[2] < opt_val1 :
				print('better than optimal validation cost, saving model ...')
				nb_epoch_no_decreasing = 0
				opt_val0 = val_loss[1]
				opt_val1 = val_loss[2]			 
				model.save_weights(output_h5)'''
			if val_loss[1] * weight + val_loss[2] < opt_total :
				print('better than optimal validation cost, saving model ...')
				nb_epoch_no_decreasing = 0
				opt_val0 = val_loss[1]
				opt_val1 = val_loss[2]
				opt_total = val_loss[1] * weight + val_loss[2]	 
				model.save_weights(output_h5)
			else : 
				nb_epoch_no_decreasing += 1
			i += 1
			if i >= n_batches:
				break
			if nb_epoch_no_decreasing > 10 * n_batches : 
				print '10 epochs no updating, stop trainning.'
				break_while = True
				break
		if break_while :
			break

	print 'begin to test...'
	print '----Loading validation model...'
	model = model_from_json(json.load(output_json))
	model.load_weights(output_h5)
	index_word = {i: w for w, i in word_index.items()}
	index_word[0] = '.'
	index_word[vocab_size] = ''
	
	size = dp.getSplitSize('test', ofwhat='images')
	t2i = np.zeros((5, size))
	i2t = np.zeros((size, size*5))
	
	for pos in range(5):
		image_test, text_test = dp.getImageSentencePairs('test', pos, reverse=reverse_sentence)
		text_test = tokenizer.texts_to_sequences(text_test)
		text_test = pad_sequences(text_test, maxlen=sentence_maxlen, padding='post')
		text_test = np.hstack([vocab_size * np.ones((text_test.shape[0], 1), dtype=text_test.dtype), text_test[:, :-1]])
		[output_encoder, _] = model.predict([image_test, text_test])
		d = output_encoder.shape[1]/2
		image_feature = output_encoder[:, :d]
		text_feature = output_encoder[:, d:]
		pdist = cdist(text_feature, image_feature, 'cosine')
		t2i[pos, :] = text2image(pdist)
		i2t[:, pos*size:(pos+1)*size] = pdist.T
	
	t2i_rank = t2i
	i2t_rank = image2text(i2t)
	
	t2i_median = np.median(t2i_rank)
	t2i_R1 =  np.sum(t2i_rank<=1)/float(50)
	t2i_R5 = np.sum(t2i_rank<=5)/float(50)
	t2i_R10 = np.sum(t2i_rank<=10)/float(50)
	
	i2t_median = np.median(i2t_rank)
	i2t_R1 = np.sum(i2t_rank<=1)/float(10)
	i2t_R5 = np.sum(i2t_rank<=5)/float(10)
	i2t_R10 = np.sum(i2t_rank<=10)/float(10)
	print 'image retrival results:'
	print 'mean rank: %f, median rank: %f, R@1: %f, R@5: %f, R@10: %f\n' %(np.mean(t2i_rank), t2i_median, t2i_R1, t2i_R5, t2i_R10)
	print 'image annotation results:'
	print 'mean rank: %f, median rank: %f, R@1: %f, R@5: %f, R@10: %f' %(np.mean(i2t_rank), i2t_median, i2t_R1, i2t_R5, i2t_R10)

	print 'begin to generate sentences...'
	lstm = model.get_layer('lstm_decoder')
	new_lstm = LSTM(lstm.output_dim, weights=lstm.get_weights(), activation='tanh', return_sequences=False, name='lstm')
	softmax = model.get_layer('softmax_out').layer 
	model_decoder = Model(model.input, softmax(new_lstm(lstm.input)))

	all_candidates = []
	max_images = dp.getSplitSize('test', ofwhat='images')
	beam_size = 20
	for i, img in enumerate(dp.iterImages(split = 'test')):
		img_feat = img['feat'].reshape(1,-1)
		
		samples, scores = beamsearch(img_feat, model_decoder, k=beam_size, vocab_size = vocab_size, maxlen = sentence_maxlen)
		candidate = ' '.join([index_word[ix] for ix in samples[0] if ix > 0]) 
		all_candidates.append(candidate)
		if i%100 == 0 :
			print '%d images have been processed ... '%i

	print 'writing intermediate files into eval/'
	open('eval/' + dataset + '_' + str(model_number), 'w').write('\n'.join(all_candidates))
	
	
	
	
	return opt_val0, opt_val1, t2i_median, t2i_R1, t2i_R5, t2i_R10, i2t_median, i2t_R1, i2t_R5, i2t_R10
		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print '******--------- Training the hybrid model of LSTM decoder and kiros loss  ------*******'
	# global setup settings, and checkpoints
	parser.add_argument('-db', '--dataset', dest='dataset', type=str, default='flickr8k', help='Dataset')
	parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=0.001, help='The step of gradient descent')
	parser.add_argument('--bs', dest='batch_size', type=int, default=100, help='Batch Size')
	parser.add_argument('--nb_epochs', dest='nb_epochs', type=int, default=50, help='Number of Epochs')
	parser.add_argument('--latent_space_dim', dest='latent_space_dim', type=int, default=300, help='Latent Space dimension for image and text')
	parser.add_argument('--hidden_state_decoder', dest='hidden_state_decoder',  type=int, default=500, help='Hidden state dimension for LSTM.')
	parser.add_argument('--sentence_maxlen', dest='sentence_maxlen',  type=int, default=30, help='Maximum sentence length.')
	parser.add_argument('--vocab_size', dest='vocab_size',  type=int, default=3000, help='Vocabulary size during the train')
	parser.add_argument('--embed_size', dest='embed_size',  type=int, default=300, help='Embedding size of text, should be 50, 100, 200, 300')
	parser.add_argument('--dropout', dest='dropout',  type=bool, default=False, help='Whether to use dropout or not')
	parser.add_argument('-write_to_csv', dest='write_to_csv', type=bool, default=True, help='whether note the information to the csv file, by default is True')
	parser.add_argument('-seed', dest='random_seed', type=int, default=1234, help='Seed to repeat the trainning')
	parser.add_argument('--loss_weight', dest='weight', type=int, default=30, help='Loss weight for Kiros loss')
	
	
	
	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	
	if params['write_to_csv'] : 
		result = copy.copy(params)
		del result['write_to_csv']
		del result['nb_epochs']
		
	
	csv_file = os.path.abspath( 'result.csv')
	if not os.path.exists(csv_file) :
		model_number = 0
	else :
		previous_result  = pd.read_csv('result.csv', index_col=0)
		model_number = len(previous_result)

	print 'This is the %dth model, the list of parameters : '%(model_number + 1)
	print json.dumps(params, indent = 2)

	
	kiros_loss, ppl_loss, t2i_median, t2i_R1, t2i_R5, t2i_R10,i2t_median, i2t_R1, i2t_R5, i2t_R10 = train_model(random_seed = params['random_seed'], 
																												dataset = params['dataset'], 
																												batch_size = params['batch_size'], 
																												nb_epochs = params['nb_epochs'], 
																												latent_space_dim = params['latent_space_dim'], 
																												hidden_state_decoder = params['hidden_state_decoder'], 
																												sentence_maxlen = params['sentence_maxlen'], 
																												embed_size = params['embed_size'], 
																												vocab_size = params['vocab_size'], 
																												learning_rate = params['learning_rate'], 
																												drop_out1 = params['dropout'], 
																												model_number = model_number, 
																												weight = params['weight'])
	result['kiros_loss'] = kiros_loss
	result['ppl_loss'] = ppl_loss
	result['model_number'] = model_number
	result['i2t_Median'] = i2t_median
	result['i2t_R1'] = i2t_R1
	result['i2t_R5'] = i2t_R5
	result['i2t_R10'] = i2t_R10
	result['t2i_Median'] = t2i_median
	result['t2i_R1'] = t2i_R1
	result['t2i_R5'] = t2i_R5
	result['t2i_R10'] = t2i_R10
	

	print 'Load the csv file with all the previous result...'
	if model_number == 0 :
		print 'First trainning, creat dataframe and save it after the trainning...'
		result = DataFrame(result, index = [model_number])
		result= result[['dataset','dropout','vocab_size','learning_rate', 'weight', 
						'sentence_maxlen', 'embed_size', 'hidden_state_decoder', 
						'latent_space_dim', 'batch_size', 'random_seed', 'kiros_loss', 
						'ppl_loss', 'i2t_Median', 'i2t_R1', 'i2t_R5', 'i2t_R10',
						't2i_Median', 't2i_R1', 't2i_R5', 't2i_R10', 'model_number']]
		result.to_csv('result.csv')
	else :
		print 'Writing to csv file...'
		result = DataFrame(result, index = [len(previous_result)])
		previous_result = previous_result.append(result)
		previous_result= previous_result[['dataset','dropout','vocab_size','learning_rate', 'weight', 
						'sentence_maxlen', 'embed_size', 'hidden_state_decoder', 
						'latent_space_dim', 'batch_size', 'random_seed', 'kiros_loss', 
						'ppl_loss', 'i2t_Median', 'i2t_R1', 'i2t_R5', 'i2t_R10',
						't2i_Median', 't2i_R1', 't2i_R5', 't2i_R10', 'model_number']]
		previous_result.to_csv('result.csv')
