#!/usr/bin/env python
import warnings
import argparse
import sys
from nltk.translate import bleu_score
import numpy
import progressbar
import six
import copy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers

UNK = 0
EOS = 1


def load_npz_no_strict(filename, obj):
    try:
        serializers.load_npz(filename, obj)
    except KeyError as e:
        warnings.warn(repr(e))
        with numpy.load(filename) as f:
            d = serializers.NpzDeserializer(f, strict=False)
            d.load(obj)
            

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):
    """ Basic Seq2Seq model without attention

    Encoder and decoder could have many layers
    Only the forward sequence is used (the model is not bidirectional)
    
    Parameters:
    n_layers: number of layers
    n_source_vocab: size of source vocabulary
    n_target_vocab: size of target vocabulary
    n_units: dim units of our network

    """
    
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():

            # EmbedID is a ID to word embed translation
            # (a lookup table)
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            # Our encoder: a N layered LSTM
            # (0.1 is the dropout ratio in output)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)

            # Our decoder: another N layered LSTM
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)

            # Transform the LSTM output of the decoder to a vector
            # of vocab size
            # (before the softmax to obtain the prob of each word)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        # Inverting the input sequence
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], 'i')
        # Output words are shifted by one at the input in the decoder)
        # 
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        # Our target words (what our model expects at every step)
        # We expect the target word at every step
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder (initial state).
        # Do the magic here!
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        # os is the output state. Now it calculates the prob of each word
        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)

        # obtain the loss at every step and sum the contributions of all
        # the sequences in the batch
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)

        # Return the loss (loss will be used by the updater to obtain
        # the gradient and update the network parameters
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            # zeroing initial target words
            ys = self.xp.full(batch, EOS, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                # picking argmax word
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

    
def stoch_translate(model, xs, max_length=100):
    """ Stochastic sampling 

    model: our seq2seq model
    xs: input sequence
    max_length: maximum length of sequence

    """
    batch = len(xs)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        xs = [x[::-1] for x in xs]
        exs = sequence_embed(model.embed_x, xs)
        h, c, _ = model.encoder(None, None, exs)
        # zeroing initial target words
        ys = model.xp.full(batch, EOS, 'i')
        result = []
        for i in range(max_length):
            eys = model.embed_y(ys)
            eys = F.split_axis(eys, batch, 0)
            h, c, ys = model.decoder(h, c, eys)
            cys = F.concat(ys, axis=0)
            wy = model.W(cys)
            # picking a word with a multinomial
            probs = F.softmax(wy)
            probs = probs - model.xp.finfo(model.xp.float32).epsneg

            # Check!
            # doing for batch
            
            #print ("PROBS DATA ", probs_data.shape)
            
            ys = []
            for j in range(batch):
                probs_data =  model.xp.reshape(probs.data[j], (probs.data.shape[1], ))
                next_sample = model.xp.random.multinomial(
                    1, probs_data).argmax(0)
                ys.append(next_sample)
             
            ys = model.xp.array(ys).reshape(batch, ).astype('i')
            result.append(ys)

    result = cuda.to_cpu(model.xp.stack(result).T)

    # Remove EOS taggs
    outs = []
    for y in result:
        inds = numpy.argwhere(y == EOS)
        if len(inds) > 0:
            y = y[:inds[0, 0]]
        outs.append(y)
    return outs


def temp_translate(model, xs, max_length=100, temperature=0.5):
    """ Sampling strategy with temperature
    
    model: our seq2seq model
    xs: input sequence
    max_length: maximum length of sequence
    temperature: temperature selects between different output selection strategies
                 argmax and a sample from a multinomial


    """
    batch = len(xs)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        xs = [x[::-1] for x in xs]
        exs = sequence_embed(model.embed_x, xs)
        h, c, _ = model.encoder(None, None, exs)
        # zeroing initial target words
        ys = model.xp.full(batch, EOS, 'i')
        result = []
        for i in range(max_length):
            eys = model.embed_y(ys)
            eys = F.split_axis(eys, batch, 0)
            h, c, ys = model.decoder(h, c, eys)
            cys = F.concat(ys, axis=0)
            wy = model.W(cys)
            if model.xp.random.binomial(1, temperature) == 1:
                # picking a word with a multinomial
                probs = F.softmax(wy)
                probs = probs - model.xp.finfo(model.xp.float32).epsneg
                ys = []
                for j in range(batch):
                    probs_data =  model.xp.reshape(probs.data[j], (probs.data.shape[1], ))
                    next_sample = model.xp.random.multinomial(
                        1, probs_data).argmax(0)
                    ys.append(next_sample)
             
                ys = model.xp.array(ys).reshape(batch, ).astype('i')
                result.append(ys)
            else:
                # picking argmax word
                ys = model.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)
                    
    result = cuda.to_cpu(model.xp.stack(result).T)

    # Remove EOS taggs
    outs = []
    for y in result:
        inds = numpy.argwhere(y == EOS)
        if len(inds) > 0:
            y = y[:inds[0, 0]]
        outs.append(y)
    return outs


def karpathy_translate(model, xs, max_length=100, temperature=0.5):
    """ A slightly modified stochastic_sampling from Karpathys blog
    (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    
    Parameters
    ----------
    model: our seq2seq model
    xs: input sequence
    max_length: maximum length of sequence
    temperature: it modifies the output probs of words
                lower temperature means the output will be closer to argmax
                (our model will be more confident in their choice => more boring)
                higher temperature means the output will be more diverse but sometimes
                it will be not very consistent (in terms of grammar)

    """
        
    batch = len(xs)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        xs = [x[::-1] for x in xs]
        exs = sequence_embed(model.embed_x, xs)
        h, c, _ = model.encoder(None, None, exs)
        # zeroing initial target words
        ys = model.xp.full(batch, EOS, 'i')
        result = []
        for i in range(max_length):
            eys = model.embed_y(ys)
            eys = F.split_axis(eys, batch, 0)
            h, c, ys = model.decoder(h, c, eys)
            cys = F.concat(ys, axis=0)
            wy = model.W(cys)
            # scale output by temperature
            wy = wy/temperature
                
            probs = F.softmax(wy)
            probs = probs - model.xp.finfo(model.xp.float32).epsneg

            ys = []
            for j in range(batch):
                probs_data =  model.xp.reshape(probs.data[j], (probs.data.shape[1], ))
                next_sample = model.xp.random.multinomial(
                    1, probs_data).argmax(0)
                ys.append(next_sample)
             
            ys = model.xp.array(ys).reshape(batch, ).astype('i')
            result.append(ys)
                    
    result = cuda.to_cpu(model.xp.stack(result).T)

    # Remove EOS taggs
    outs = []
    for y in result:
        inds = numpy.argwhere(y == EOS)
        if len(inds) > 0:
            y = y[:inds[0, 0]]
        outs.append(y)
    return outs


def beam_search_translate(model, xs, k_beam=4, max_length=100, use_unk=False):
    """ A basic beam search implementation

    Parameters:
    -----------
    model: our seq2seq model
    xs: our source sequence of ids
    k_beam: size of beam search
    max_length: maximum length of sequence
    use_unk: if True, UNK could be used as output word
             if False, an alternative word should be used instead

    """
    
    sample = []
    sample_score = []
    
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = model.xp.zeros(live_k).astype('float32')
    hyp_states = []
    
    batch = len(xs)
    
    assert(batch) == 1, "batch must be 1 in this implementation"
    
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        # obtaining initial state in encoder
        xs = [x[::-1] for x in xs]
        
        exs = sequence_embed(model.embed_x, xs)
        
        # h and c are #layers x #batch x # dim
        h, c, _ = model.encoder(None, None, exs)
    
        # repeat encoder state till full beam is reached
        h = chainer.Variable(h.data.repeat(k_beam, axis=1))
        c = chainer.Variable(c.data.repeat(k_beam, axis=1))
    
        # zeroing initial target words
        ys = model.xp.full(k_beam, EOS, 'i')
    
        for i in range(max_length):
            live_batch = len(ys)
            
            eys = model.embed_y(ys)
            eys = F.split_axis(eys, live_batch, 0)
            
            h, c, ys = model.decoder(h, c, eys)
            cys = F.concat(ys, axis=0)
            wy = model.W(cys)
        
            probs = F.softmax(wy)
            probs_data = probs.data - model.xp.finfo(model.xp.float32).epsneg

            cand_scores = hyp_scores[:, None] - model.xp.log(probs_data[:live_k, :])
            cand_flat = cand_scores.flatten()
               
            # TODO filter UNK words here (before ranking)
            voc_size = probs_data.shape[1]
            
            if not use_unk:
                for xx in range(int(len(cand_flat) / int(voc_size))):
                      cand_flat[voc_size * xx + UNK] = 1e20
            
            ranks_flat = cand_flat.argsort()[:(k_beam-dead_k)]
            

            trans_indices = ((ranks_flat / voc_size)).astype('int32')
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = model.xp.zeros(k_beam-dead_k).astype('float32')
            new_hyp_states = []
            
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy((h[:,ti,:],c[:,ti,:])))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            
            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == EOS:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k_beam:
                break
                                      
            # Next state
            # state is #layers x #batch x #dim
            ys = numpy.array([w[-1] for w in hyp_samples]).astype('i')
            h = F.stack([hyp_states[k][0] for k in range(len(hyp_states))], axis=1)
            c = F.stack([hyp_states[k][1] for k in range(len(hyp_states))], axis=1)
            
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score
            

def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})

        
class CalculateBleuSnapshot(chainer.training.Extension):
    """ Calculate Bleu and generate snapshot if the bleu score is better
     than our previous model """
    
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, optimizer, test_data, key, batch=100, device=-1,
            max_length=100, saveto=None):
        
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.saveto = saveto
        self.best_bleu = sys.float_info.min
        self.optimizer = optimizer

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            # This is BLEU notation
            # hypotheses are our predicted sentences (model translation)
            # references are the real samples (our targets)
            # Bleu score measure the overlap among hypotheses and references
            # (higher is better)
            references = []
            hypotheses = []

            # loop over a batch of samples size
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])

                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                # Obtain the translation of source sentences
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)

        if bleu > self.best_bleu:
            # save snapshot to folder if the bleu is improved
            if self.saveto is not None:
                serializers.save_npz("{}/model.npz".format(self.saveto),
                                     self.model)
                serializers.save_npz("{}/optimizer.npz".format(self.saveto),
                                     self.optimizer)
                
            self.best_bleu = bleu
            
        chainer.report({self.key+"/best": self.best_bleu})
        chainer.report({self.key: bleu})
        

def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK) for w in words], 'i')
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evaluate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--saveto', default='saveto',
                        help='folder to save the snapshot (model, optimizer)')
    
    args = parser.parse_args()

    # load source and target vocabs
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    # load training data
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)

    # Check source and target has the same number of samples
    assert len(train_source) == len(train_target)

    # Filter sentences that did not has the apropriate lengths
    # in source or target (min, max)
    train_data = [(s, t)
                  for s, t in six.moves.zip(train_source, train_target)
                  if args.min_source_sentence <= len(s)
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t)
                  <= args.max_source_sentence]
    # Stats: Calculate unknown words (due to our limited vocabularies)
    train_source_unknown = calculate_unknown_ratio(
        [s for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
        [t for _, t in train_data])

    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_data))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    # inverted vocabs (for idx to word translation)
    # internally our model works with indexes, not with words
    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Build the model (a standard seq2seq model)
    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # load the model (if resuming from previous checkpoint)
    if args.resume is not '':
        load_npz_no_strict("{}/model.npz".format(args.resume), model)
        load_npz_no_strict("{}/optimizer.npz".format(args.resume), optimizer)

    # An iterator is basically functionallity for extracting batch of
    # samples from a dataset.
    # The model, as usual, is trained in minibatches of size batchsize
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # The updater updates our network parameters (aka. our optimizer)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # Extensions are extra functionality
    # LogReport builds a log file with accumulated stats
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))

    # PrintReport selects the variables to print to the file
    # (if they exist in that interation)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'validation/main/bleu/best',
         'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))

    # If we are using validation data add extra extensions
    if args.validation_source and args.validation_target:
        # Load validation data
        test_source = load_data(source_ids, args.validation_source)
        test_target = load_data(target_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' %
              (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        # Generate samples from the test data and measure the error obtained
        # (in model.translate)
        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[numpy.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source : ' + source_sentence)
            print('#  result : ' + result_sentence)
            print('#  expect : ' + target_sentence)

        # Generate samples 
        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))

        # Obtains the bleu error of samples
        # Firstly, a sample is generated
        # Secondly, the bleu score is obtained for a set of samples.
        # Bleu score is a metric that measures the word overlap among sentences
        # (typically used in translation)
        trainer.extend(
            CalculateBleuSnapshot(
                model,
                optimizer,
                test_data, 'validation/main/bleu',
                device=args.gpu,
                saveto=args.saveto),
            trigger=(args.validation_interval, 'iteration'))
            
    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
