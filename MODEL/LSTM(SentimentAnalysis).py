
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import spacy
import sklearn as sl


# In[40]:


word_embeddings = spacy.load('en', vectors='glove.6B.300d.txt')


# In[59]:


def sequence_to_data(seq, max_len=None):    ####Converting sequence to data basically convertig words to vectors
    seq = unicode(seq, 'utf-8')
    data = [word_embeddings(ix).vector for ix in seq.split()]
    if max_len == None:
        max_len = len(data)
        
    data_mat = np.zeros((1, max_len, 384))
    for ix in range(min(max_len, len(data))):
        data_mat[:, ix, :] = data[ix]

    return data_mat

def seq_data_matrix(seq_data, max_len=None):  ####Now Concating different sentences and converting to a matrix
    data = np.concatenate([sequence_to_data(ix, max_len) for ix in seq_data], axis=0)
    return data


# In[42]:


df = pd.read_csv("/Users/adityakumar/Desktop/dataset/final.csv")#loading dataset


# In[43]:


df.head()


# In[44]:


df = df.drop('Unnamed: 0', axis=1)
df = df.dropna()
df = df.reset_index(drop=True)


# In[45]:


df.head()


# In[46]:


df['len'] = ' '   ###Adding column 'len' for no. of words in preprocessed tweet


# In[47]:


for ix in range(df.shape[0]):    ###Now assigning values in cells of column of 'len'
    a = len(str(df['preprocessed tweet'].loc[ix]).split())
    df.loc[ix, 'len'] = a


# In[48]:


bucket_sizes = [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 35]]

def assign_bucket(x):       ###making buckets of different sizes and assigning bucket to tweets according to their 'len'
    for bucket in bucket_sizes:
        if x>=bucket[0] and x<=bucket[1]:
            return bucket_sizes.index(bucket)
    return len(bucket_sizes)-1


# In[49]:


df['bucket'] = df.len.apply(assign_bucket)
df.head()


# In[50]:


df = df.sort_values(by=['bucket'])
df.head()


# In[51]:


df['preprotweet'] = df['preprocessed tweet']


# In[52]:


df = df.drop('preprocessed tweet', axis=1)


# In[53]:


df.head()


# In[54]:


def make_batch(data, batch_size=10, gpu=False):# making batches to pass in model during training
    for bx in range(len(bucket_sizes)):
        bucket_data = df[(df.bucket==bx)].reset_index(drop=True)
        
        start = 0
        stop = start + batch_size
        while start < bucket_data.shape[0]:
            seq_len = bucket_sizes[bx][1]
            section = bucket_data[start:stop]
            xdata = seq_data_matrix(section.preprotweet, max_len=seq_len)
            ydata = section.label
            if gpu == True:
                yield Variable(torch.FloatTensor(xdata).cuda(), requires_grad=True), Variable(torch.LongTensor(ydata)).cuda()
            else:
                yield Variable(torch.FloatTensor(xdata), requires_grad=True), Variable(torch.LongTensor(ydata))
            
            start = stop
            stop = start + batch_size
    


# ### Model of LSTM

# In[66]:


class SentModel(nn.Module):
    def __init__(self, in_shape=None, out_shape=None, hidden_shape=None):
        super(SentModel, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_shape = hidden_shape
        self.n_layers = 1
        
        self.rnn = nn.LSTM(
                        input_size = self.in_shape,
                        hidden_size = self.hidden_shape,
                        num_layers = self.n_layers,
                        batch_first = True
        )
        self.lin = nn.Linear(self.hidden_shape, 64)
        self.dropout = nn.Dropout(0.42)
        self.out = nn.Linear(64, self.out_shape)
        
        
    def forward(self, x, h):
        r_out, h_state = self.rnn(x, h)
        last_out = r_out[:, -1, :]
        y = f.tanh(self.lin(last_out))
        y = self.dropout(y)
        y = f.softmax(self.out(y))
        return y
    
    def predict(self, x):
        h_state = self.init_hidden(1)    
        x = sequence_to_data(x)
        pred = self.forward(torch.FloatTensor(x), h_state)
        return pred
    
    def get_embedding(self, x):
        h_state = self.init_hidden(1, gpu=False)
        
        x = sequence_to_data(x)
        r_out, h = self.rnn(torch.FloatTensor(x), h_state)
        last_out = r_out[:, -1, :]
        
        return last_out.data.numpy()
    
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_shape)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_shape)))


# In[67]:


model = SentModel(in_shape=384, hidden_shape=256, out_shape=2)

print(model)


# In[57]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()


# In[60]:


for epoch in range(50):
    total_loss = 0
    N = 0
    for step, (b_x, b_y) in enumerate(make_batch(df, batch_size=200)):
        bsize = b_x.size(0)
        
        h_state = model.init_hidden(bsize)

        pred = model(b_x, h_state)
        loss = criterion(pred, b_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        N += 1.0
        if step%2 == 0:
            print('Loss: {} at Epoch: {} | Step: {}'.format(loss, epoch, step))
        
    print("Overall Average Loss: {} at Epoch: {}".format(total_loss / float(N), epoch))
    
 

torch.save(model.state_dict(), "model_256h_epoch_{}.ckpt".format(epoch))


# In[61]:


torch.save(model.state_dict(), "model_256h_lstm.ckpt")


# In[62]:


model.eval()


# In[69]:


model.load_state_dict(torch.load("model_256h_lstm.ckpt"))


# In[78]:


model.predict('indianlovestori never let anyon hurt throw acid face make point india love acid')

