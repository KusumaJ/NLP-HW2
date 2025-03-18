import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    # TODO: complete for part 3
    self.embedding = Embedding(num_embeddings=word_types, embedding_dim=128)
    self.hiddenLayer = Linear(768,128) 
    self.outputLayer = Linear(128,outputs)

  def forward(self, inputs):

    # TODO: complete for part 3
    embedded = self.embedding(inputs)
    inputs = embedded.view([embedded.size()[0],embedded.size()[1]*embedded.size()[2]]) # batch_size, 768
    hidden = self.hiddenLayer(inputs)
    hidden = relu(hidden)
    output = self.outputLayer(hidden)
    return output

def train(model, loader): 

  loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch
 
    predictions = model(torch.LongTensor(inputs.long()))

    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")
  # In Prof's experiments, after 5 epochs reached a training loss of < 0.31 and a training accuracy of about 0.90.
  # 5 epoch i got Training loss epoch: Training loss epoch: 0.30688479310471745,   Accuracy: 0.9026153683662415

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary file {}.".format(WORD_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 
