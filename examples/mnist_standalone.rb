# coding: utf-8
require 'ruby_brain'
require 'ruby_brain/dataset/mnist/data'


dataset = RubyBrain::DataSet::Mnist::data
training_dataset = dataset.first
test_dataset = dataset.last


training_dataset.keys # => [:input, :output]
test_dataset.keys     # => [:input, :output]

training_dataset[:input].size       # => 60000
training_dataset[:input].first.size # => 784

training_dataset[:output].size       # => 60000
training_dataset[:output].first.size # => 10

test_dataset[:input].size       # => 10000
test_dataset[:input].first.size # => 784

test_dataset[:output].size       # => 10000
test_dataset[:output].first.size # => 10


# use first 5000 pictures for training
NUM_TRAIN_DATA = 5000
training_input = training_dataset[:input][0..(NUM_TRAIN_DATA-1)]
training_supervisor = training_dataset[:output][0..(NUM_TRAIN_DATA-1)]

# use all pictures within test_dataset
test_input = test_dataset[:input]
test_supervisor = test_dataset[:output]

# network structure [784, 50, 10]
network = RubyBrain::Network.new([training_input.first.size, 50, training_supervisor.first.size])
# learning rate is 0.7
network.learning_rate = 0.7
# initialize network
network.init_network

network.learn(training_input, training_supervisor, max_training_count=100, tolerance=0.0004, monitoring_channels=[:best_params_training])

### turn on this snippet to print pictures as ascii art. 
# 
# test_input.each_with_index do |input, i|
#   input.each_with_index do |e, j|
#     print(e > 0.3 ? 'x' : ' ')
#     puts if (j % 28) == 0
#   end
#   puts
#   supervisor_label = test_supervisor[i].index(test_supervisor[i].max)
#   predicated_output = network.get_forward_outputs(input)
#   predicated_label = predicated_output.index(predicated_output.max)
#   puts "test_supervisor: #{supervisor_label}"
#   puts "predicate: #{predicated_label}"
#   results << (supervisor_label == predicated_label)
#   puts "------------------------------------------------------------"
# end

puts "accuracy: #{results.count(true).to_f/results.size}"
