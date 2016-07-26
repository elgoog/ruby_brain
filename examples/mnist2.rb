require 'ruby_brain'
require 'ruby_brain/dataset/mnist/data'

NUM_TRAIN_DATA = 5000
NUM_TEST_DATA = 500

dataset = RubyBrain::DataSet::Mnist::data

training_input = dataset[:input][0..(NUM_TRAIN_DATA-1)]
training_supervisor = dataset[:output][0..(NUM_TRAIN_DATA-1)]

# test_input = dataset[:input][NUM_TRAIN_DATA..(NUM_TRAIN_DATA+NUM_TEST_DATA-1)]
# test_supervisor = dataset[:output][NUM_TRAIN_DATA..(NUM_TRAIN_DATA+NUM_TEST_DATA-1)]
test_input = dataset[:input][NUM_TRAIN_DATA..-1]
test_supervisor = dataset[:output][NUM_TRAIN_DATA..-1]

network = RubyBrain::Network.new([dataset[:input].first.size, 50, dataset[:output].first.size])
# network.learning_rate = 0.7
network.init_network
network.load_weights_from_yaml_file(File.dirname(__FILE__) + '/../best_weights_1469044985.yml')

### You can initializes weights by loading weights from file if you want.
# network.load_weights_from_yaml_file('path/to/weights.yml.file')

# network.learn(training_input, training_supervisor, max_training_count=100, tolerance=0.0004, monitoring_channels=[:best_params_training])

### You can save weights into a yml file if you want.
# network.dump_weights_to_yaml('path/to/weights.yml.file')


class Array
  def argmax
    max_i = 0
    max_val = self[max_i]
    self.each_with_index do |v, i|
      if v > max_val
        max_val = v
        max_i = i
      end
    end
    return max_i
  end
end

results = []
test_input.each_with_index do |input, i|
  ### You can see test input, label and predicated lable in standard out if you uncomment in this block 

  input.each_with_index do |e, j|
    print(e > 0.3 ? 'x' : ' ')
    puts if (j % 28) == 0
  end
  puts
  supervisor_label = test_supervisor[i].argmax
  predicated_label = network.get_forward_outputs(test_input[i]).argmax
  puts "test_supervisor: #{supervisor_label}"
  puts "predicate: #{predicated_label}"
  results << (supervisor_label == predicated_label)
  puts "------------------------------------------------------------"
end

puts "accuracy: #{results.count(true).to_f/results.size}"



### you can do above procedure simply by using Trainer

# training_option = {
#   learning_rate: 0.5,
#   max_training_count: 50,
#   tolerance: 0.0004,
#   # initial_weights_file: 'weights_3_30_10_1429166740.yml',
#   # initial_weights_file: 'best_weights_1429544001.yml',
#   monitoring_channels: [:best_params_training]
# }

# RubyBrain::Trainer.normal_learning([dataset[:input].first.size, 50, dataset[:output].first.size],
#                                    training_input, training_supervisor,
#                                    training_option)



