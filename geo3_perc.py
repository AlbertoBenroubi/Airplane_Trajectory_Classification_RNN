import functools
import sets
import tensorflow as tf
import glob
import sys
import numpy as np

train_i="train_i2/*.txt"
train_l="train_l2/*.txt"

training_iters=4770

def send_email(user, pwd, recipient, subject, body):
    import smtplib

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print 'successfully sent the mail'
    except:
        print "failed to send mail"

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper




def read_data_files_train(directory_i_train,  directory_l_train, step_n):
    list_dir_i_train = glob.glob(directory_i_train)
    list_dir_l_train = glob.glob(directory_l_train)
    x_data = []
    y_data = []
    for trainfile in list_dir_i_train:
        if trainfile.endswith("batch"+str(step_n)+".txt"):
       
            print("Loading file " + trainfile)
            x_data=np.loadtxt(trainfile)
            
            x_data=np.array(x_data)
        

    for trainfile in list_dir_l_train:
        if trainfile.endswith("batch"+str(step_n)+".txt"):
   
            print("Loading file " + trainfile)
            y_data=np.loadtxt(trainfile)
            y_data=np.array(y_data)
           

    return x_data,y_data



class VariableSequenceClassification:

    def __init__(self, data, target, num_hidden=450, num_layers=1):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.state = None
        self.prediction
        self.confusion
        self.error
        self.accuracy
        self.optimize

    @lazy_property
    def confusion(self):
    	decoded_target=tf.argmax(self.target,axis=1)
    	decoded_prediction=tf.argmax(self.prediction,axis=1)
        confusion=tf.confusion_matrix(decoded_target,decoded_prediction,num_classes=12)
        return confusion

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self._num_hidden, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.sigmoid)
		
        output, _state = tf.nn.dynamic_rnn(
            lstm_cell,
            data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        #print output.shape
        
        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        self.state=_state
        #print self.state
        print prediction.shape
        print target.shape
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        #print(mistakes)
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def accuracy(self):
        correct_pred = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        #print output.shape
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
batch_size=45
step = 0
n_classes=12
n_inputs = 3   # MNIST data input (img shape: 28*28)
max_length = 2105  # time steps

if __name__ == '__main__':

    x_final,y_final=read_data_files_train(train_i,train_l,step+1)
    x_final = x_final.reshape([batch_size, max_length, n_inputs])

    y_final=y_final.reshape([batch_size, n_classes])
    _, max_size, features = x_final.shape

    num_classes = y_final.shape[1]
    data = tf.placeholder(tf.float32, [None, max_size, features])
    target = tf.placeholder(tf.float32, [None, num_classes])
    state=tf.zeros
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,"output/model.ckpt")
    confusion_all=np.zeros((12,12),dtype=np.int)
    accuracy_sum=0
    for k in range(20):
        test_x,test_y=read_data_files_train(train_i,train_l,107+k)
        test_x=test_x.reshape([batch_size, max_length, n_inputs])
        test_y=test_y.reshape([batch_size, n_classes])
        accuracy = sess.run(model.accuracy, {data: test_x, target: test_y})
        #length= sess.run(model.length, {data: test_x, target: test_y})
        #print(length)
        #target_= sess.run(model.target, {data: test_x, target: test_y})
        #print(target_[0])
        #prediction_= sess.run(model.prediction, {data: test_x, target: test_y})
        #print(prediction_[0])
        confusion= sess.run(model.confusion, {data: test_x, target: test_y})
        print(confusion)
        accuracy_sum=accuracy_sum+accuracy
    	confusion_all=confusion_all+confusion
    	print(confusion_all)
    mean_accuracy=accuracy_sum/20
    print('accuracy {:3.1f}%'.format(100 * mean_accuracy))
  
   