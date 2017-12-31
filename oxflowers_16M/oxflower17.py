import tflearn.datasets.oxflower17 as oxflower17
import numpy
class Dataset():
    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
    def read_data(self):
        m=0
        n=0
        X, Y = oxflower17.load_data(one_hot=True)
        self.train_data = numpy.zeros([935,224,224,3],dtype="float32")
        self.train_label = numpy.zeros([935,17],dtype="float32")
        self.validate_data = numpy.zeros([425,224,224,3],dtype="float32")
        self.validate_label = numpy.zeros([425,17],dtype="float32")

        for i in range(0,1360):
            if i % 80 < 55:
                numpy.copyto(self.train_data[m],X[i])
                numpy.copyto(self.train_label[m],Y[i])
                m += 1
            else:
                self.validate_data[n] = numpy.copy(X[i])
                self.validate_label[n] = numpy.copy(Y[i])
                n += 1
        self._num_examples = self.train_data.shape[0]
        return self.train_data,self.train_label,self.validate_data,self.validate_label

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples,dtype="int32")
            numpy.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.train_label = self.train_label[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_data[start:end],self.train_label[start:end]


