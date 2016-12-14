"""
Dynamic Memory Networks
The implementation of _______
"""

from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import rnn_cell
from tqdm import tqdm

class  DMN_Model(object):
    '''This is a beautiful model I like very much
    '''

    def __init__(self, config, embeddings):
        '''We use the denotes for subsequent comments:
        N: the size of mini-batch = batch_size
        H: the size of hidden state = hidden_size
        D: the size of word embedding = embed_size
        F: the max facts size
        S: the max words size of the sentence
        Q: the max words size of the question
        M: the max memory updating passes
        V: the vocabulary size
        '''

        self.N = config.batch_size
        self.H = config.hidden_size
        self.Ha = config.attention_hidden_size
        self.D = config.embed_size
        self.F = config.max_facts_len
        self.S = config.max_sentence_len
        self.Q = config.max_question_len
        self.M = config.max_memory_pass
        self.V = config.vocab_size

        self.epoch_size = config.epoch_size
        self.weight_decay = config.weight_decay
        self.learning_rate = config.learning_rate
        self.initial_embeddings = embeddings
        self.save_prefix = config.save_prefix
        self.save_period = config.save_period
        self.train_period = config.train_period
        self.dev_period = config.dev_period
        self.print_period = config.print_period

        
        with tf.variable_scope('DMN') as scope:
            print('Building graph...  add_variables')
            self.add_variables()
            print('Building graph...  add_placeholders')
            self.add_placeholders()
            print('Building graph...  add_embeddings')
            self.add_embeddings()
            print('Building graph...  add_input_question_module')
            self.add_input_question_module()
            print('Building graph...  add_episode_memory_module')
            self.add_episode_memory_module()
            print('Building graph...  add_answer_module')
            self.add_answer_module()
            print('Building graph...  add_total_loss')
            self.add_total_loss()
            print('Building graph...  add_train_op')
            self.add_train_op()
            print('Building graph...  add_test_op')
            self.add_test_op()
            print('Builded')

        self.saver = tf.train.Saver() # Use to save model
        self.train_summary_writter = tf.train.SummaryWriter(config.train_summary_dir) 
        self.dev_summary_writter = tf.train.SummaryWriter(config.dev_summary_dir)


    def add_variables(self):

        # add the weight to weight decay regulization
        def get_variable(name, shape, dtype):
            weight = tf.get_variable(name, shape=shape, dtype=dtype)
            tf.add_to_collection('weight_decay', tf.nn.l2_loss(weight))
            return weight

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # with tf.variable_scope('Attention'):
        self.Wq = get_variable('Wq', shape=[self.H,self.H], dtype=tf.float32) # (H, H)
        self.Wm = get_variable('Wm', shape=[self.H,self.H], dtype=tf.float32) # (H, H)
        self.W1 = get_variable('W1', shape=[5*self.H+4, self.Ha], dtype=tf.float32) # (5*H+4, h1)
        self.b1 = tf.get_variable('b1', shape=[self.Ha], dtype=tf.float32) # (h1, )
        self.W2 = get_variable('W2', shape=[self.Ha, 1], dtype=tf.float32) # (h1, 1)
        self.b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32) #(1, )
        
        # with tf.variable_scope('Answer_Module'):
        self.Wa = get_variable('Wa', shape=[self.H, self.V], dtype=tf.float32) # (H,V)

    def add_placeholders(self):
        """Generate placeholder variables

        self.inputs_placeholder: Input Module's paragraph placeholder shape of
                                (batch_size,None), type of int32
        self.questions_placeholder: Question Module's question placeholder shape of
                                (batch_size,None), type of int32
        """

        # shape of (batch_size, sentence_len)
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=[self.N, self.S], name='Paragraph') #(N, S)
        self.fact_masks_placeholder = tf.placeholder(tf.bool, shape=[self.N,self.S],name='Fact_Mask') #(N,S)
        self.questions_placeholder = tf.placeholder(tf.int32, shape=[self.N, self.Q], name='Question') #(N,Q)
        self.answers_placeholder = tf.placeholder(tf.int32, shape=[self.N], name='Answer') #(N, )
        self.is_training = tf.placeholder(tf.bool, shape=[]) # (,)
        

    def add_embeddings(self):
        """Take the word embedding from the lookup table.
        Hint: tf.nn.embedding_lookup only available in cpu now

        The Input Module and The Question Module use the same word embedding lookup table.
        """

        with tf.device('/cpu:0'):
            with tf.variable_scope('Embedding_Layer'):
                embeddings = tf.Variable(self.initial_embeddings,name = 'Embeddings')
                self.input_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
                self.question_embeddings = tf.nn.embedding_lookup(embeddings, self.questions_placeholder)

    def add_input_question_module(self):
        """ Generate the Input module describe in the paper above.
        """

        # we use the GRU here
        gru = rnn_cell.GRUCell(self.H)

        with tf.variable_scope('Input_Layer') as scope:

            with tf.name_scope('Input_Module'):
                # input module
                input_states, _ = tf.nn.dynamic_rnn(gru, self.input_embeddings,dtype = tf.float32) # (N,S,H)
                # input mask to get the facts
                facts = []
                for n in range(self.N):
                    example_fact = tf.boolean_mask(input_states[n,:,:], self.fact_masks_placeholder[n,:]) #(?,H)
                    # padding the fact to a static length
                    paddings = tf.zeros([(self.F-tf.shape(example_fact)[0]), self.H], name='paddings')
                    example_fact = tf.concat(0, [example_fact, paddings])
                    facts.append(example_fact)
                facts = tf.pack(facts)

            # input module and question module use the same parameters    
            scope.reuse_variables()

            with tf.name_scope('Question_Module'):
                # question module
                _, question_state = tf.nn.dynamic_rnn(gru,self.question_embeddings,dtype = tf.float32)

        self.facts = facts # (N,F,H)
        self.question = question_state # (N,H)

    def attention_machanism(self, c, q, m):
        '''Attention machanism

        Arg:
            c : The facts at one time step shape of (N, H)
            q : The question final state shape of (N, H)
            m : The memory at one pass shape of (N, H)
        '''

        def bilinear_matmul(x,W,y):
            result = tf.batch_matmul(tf.expand_dims(tf.matmul(x,W),-2),tf.expand_dims(y,-1)) # (N,1,1)
            return tf.squeeze(result,[-1]) # (N,1)
            
        # The fators
        fators = tf.concat(1, [c, m, q, c*q, c*m, 
                                        tf.expand_dims(tf.reduce_sum((c-q)**2, 1),-1),
                                        tf.expand_dims(tf.reduce_sum((c-m)**2, 1),-1), 
                                        bilinear_matmul(c,self.Wq,q),
                                        bilinear_matmul(c,self.Wm,m)]) # (N, 5*H+4)

        # The forward neural network
        l1 = tf.tanh(tf.matmul(fators, self.W1) + self.b1)
        l2 = tf.sigmoid(tf.matmul(l1, self.W2) + self.b2)

        return l2

    def get_episode(self,facts,question,memory):
        '''The inner RNN of Episode Memory Networks.
        We generate the episode using the final hidden state.
            
        Arg: 
            facts: The output from Input Module shape of (N,F,H)
            question: The output from Question Module shape of (N,H)
            memory : The memory to be updated shape of (N,H)
        '''
        gru = rnn_cell.GRUCell(self.H)

        c = tf.unpack(tf.transpose(facts, [1, 0, 2]), self.F) # F-len List of Tensor shape of (N,H)
        initial_state = tf.zeros([self.N, self.H])
        hidden_state = initial_state
        with tf.variable_scope('Episode') as scope:
            for ct in c: # ct shape of (N,H)
                ct.set_shape([self.N, self.H])
                gt = self.attention_machanism(ct,question,memory)
                hidden_state = gt*gru(ct, hidden_state)[0] + (1-gt)*hidden_state
                scope.reuse_variables()
                
        episode = hidden_state
        return episode

    def add_episode_memory_module(self):
        '''The outer RNN of Episode Memory Networks

        self.facts: The output from Input Module shape of (N,F,H)
        self.question: The output from Question Module shape of (N,H)
        '''

        # m0 = q refers from the paper
        memory = tf.identity(self.question)

        gru = rnn_cell.GRUCell(self.H)
        with tf.variable_scope('Memory_Module') as scope:
            for i in range(self.M):
                episode = self.get_episode(self.facts, self.question, memory)
                _, memory = gru(episode, memory) 
                scope.reuse_variables()
        
        # return the final memory for the Answer Module
        self.memory = memory        

    def add_answer_module(self):
        '''The Answer Module to output the answer using the final memory

        self.memory : The final memory shape of (N,H)
        '''
        # Use a linear layer
        self.logits = tf.matmul(self.memory, self.Wa) # (N,V)

    def add_total_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.answers_placeholder)
        self.total_loss = tf.reduce_mean(cross_entropy) + self.weight_decay * tf.add_n(tf.get_collection('weight_decay'))


    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
        self.train_op = train_op

    def add_test_op(self):
        predicts = tf.to_int32(tf.argmax(self.logits, 1)) # (N, )
        corrects = tf.equal(predicts, self.answers_placeholder) # (N, )
        
        self.accuracy = tf.reduce_mean(tf.to_float(corrects))
        self.num_corrects = tf.reduce_sum(tf.to_float(corrects))

    # The thing below should be split from this class.
    # But I can't find a good way to place them

    def load_model(self, sess):
        checkpoint = tf.train.get_checkpoint_state(self.save_prefix)
        if checkpoint is None:
            sys.stderr.write('No saved model...')
            sys.stderr.flush()
        return self.saver.restore(sess, checkpoint.model_checkpoint_path)

    def train_epoch(self, sess, train_data):
        '''Use mini-batch'''
        for batch in train_data:
            loss, global_step, _ = sess.run([self.total_loss, self.global_step, self.train_op], feed_dict=self.get_feed_dict(batch, True))
        return loss, global_step

    def train(self, sess, train_data, dev_data):
        merged_summary_op = tf.merge_all_summaries()
        print('Prepare to training...')
        for epoch_now in tqdm(range(1, self.epoch_size+1), leave=False):
            loss, global_step = self.train_epoch(sess, train_data)

            # Save model and print
            if not epoch_now % self.print_period:
                print('[Info {}]: Train loss: {}  global_step: {}'.format(epoch_now, loss,global_step))
            if not epoch_now % self.save_period : 
                print('[Saving {}]: Train loss: {}  global_step: {}'.format(epoch_now, loss,global_step))
                self.saver.save(sess, self.save_prefix, self.global_step)
            # Validation
            if not epoch_now % self.train_period:
                acc, loss, global_step = self.test(sess, train_data)
                #train_summary = sess.run([merged_summary_op])
                #self.train_summary_writter.add_summary(train_summary, global_step)
                print('[Train Validation {}]: acc: {} loss:{} global_step: {}'.format(epoch_now, acc, loss, global_step))
            if dev_data and epoch_now % self.dev_period:
                acc, loss, global_step = self.test(sess, dev_data)
                #dev_summary = sess.run(merged_summary_op)
                #self.dev_summary_writter.add_summary(dev_summary, global_step)
                print('[Dev Validation {}]: acc: {} loss:{} global_step: {}'.format(epoch_now, acc, loss, global_step))
        
    def test(self, sess, test_data):
        accs, losses = [], []
        for batch in test_data:
            acc, loss, global_step = sess.run([self.accuracy, self.total_loss,self.global_step], feed_dict=self.get_feed_dict(batch, False))
            accs.append(acc)
            losses.append(loss)
        return np.mean(accs), np.mean(losses), global_step

        return 

    def get_feed_dict(self, batch, is_train):
        return {
            self.inputs_placeholder: batch.contexts, 
            self.questions_placeholder: batch.questions, 
            self.answers_placeholder: batch.answers, 
            self.fact_masks_placeholder: batch.fact_masks, 
            self.is_training: is_train
            }