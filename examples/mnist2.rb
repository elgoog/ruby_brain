require 'ruby_brain'
require 'ruby_brain/dataset/mnist/data'

NUM_TEST_DATA = 500

dataset = RubyBrain::DataSet::Mnist::data

test_dataset = dataset.last
test_input = test_dataset[:input]
test_supervisor = test_dataset[:output]

# test_input = dataset[:input][NUM_TRAIN_DATA..(NUM_TRAIN_DATA+NUM_TEST_DATA-1)]
# test_supervisor = dataset[:output][NUM_TRAIN_DATA..(NUM_TRAIN_DATA+NUM_TEST_DATA-1)]
test_input = test_dataset[:input][NUM_TRAIN_DATA..-1]
test_supervisor = test_dataset[:output][NUM_TRAIN_DATA..-1]

network = RubyBrain::Network.new([test_input.first.size, 50, test_supervisor.first.size])
# network.learning_rate = 0.7
network.init_network

### You can initializes weights by loading weights from file if you want.
network.load_weights_from_yaml_file(File.dirname(__FILE__) + '/../best_weights_1469999296.yml')

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

