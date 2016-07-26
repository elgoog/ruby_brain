require 'yaml'

module RubyBrain::Trainer
  module_function

  def normal_learning(network_structure, input_training_set, output_training_set, options={})
    default_options = {learning_rate: 0.05, max_training_count: 5000, tolerance: 0.045, initial_weights_file: nil}
    training_options = default_options.merge(options)
    puts "===== normal learnng ====="
    pp training_options
    network = RubyBrain::Network.new(network_structure)
    network.learning_rate = training_options[:learning_rate]
    network.init_network
    if training_options[:initial_weights_file]
      puts "loading weights from #{training_options[:initial_weights_file]}"
      network.load_weights_from_yaml_file(training_options[:initial_weights_file])
    end
    network.learn(input_training_set, output_training_set, max_training_count=training_options[:max_training_count], tolerance=training_options[:tolerance])

    network.dump_weights_to_yaml("weights_#{network_structure.join('_')}_#{Time.now.to_i}.yml")

    # input_training_set.each do |inputs|
    #   puts network.get_forward_outputs(inputs).join(',')
    # end
  end

  def learn2(network_structure, input_training_set, output_training_set)
    network = RubyBrain::Network.new(network_structure)
    network.learning_rate = 0.05
    network.init_network
    network.learn2(input_training_set, output_training_set, max_training_count=5000, tolerance=0.045)

    puts network.dump_weights_to_yaml

    input_training_set.each do |inputs|
      pp network.get_forward_outputs(inputs)[0]
    end
  end

  def stack_learning(network_structure, input_training_set, output_training_set)
      ws = Array.new(network_structure.size-1, nil)
      (1..(network_structure.size-2)).each do |i|
        next_network_form = network_structure[0..i] + [network_structure[-1]]
        neuralnet = RubyBrain::Network.new(next_network_form)
        neuralnet.learning_rate = 0.05
        neuralnet.init_network
        neuralnet.overwrite_weights(ws)
        # neuralnet.learn_only_specified_layer(-1, input_training_set, output_training_set, max_training_count=1000, tolerance=0.06)
        neuralnet.learn2(input_training_set, output_training_set, max_training_count=500, tolerance=0.006)
        ws = neuralnet.get_weights_as_array
        ws[-1] = nil
      end
      neuralnet = RubyBrain::Network.new(network_structure)
      neuralnet.learning_rate = 0.05
      neuralnet.init_network
      neuralnet.overwrite_weights(ws)
      neuralnet.learn(input_training_set, output_training_set, max_training_count=2000, tolerance=0.0036)
      # neuralnet.learn2(input_training_set, output_training_set, max_training_count=1000, tolerance=0.036)

      puts neuralnet.dump_weights_to_yaml

      input_training_set.each do |inputs|
        pp neuralnet.get_forward_outputs(inputs)[0]
      end
      puts "==================================================================================================="
      input_training_set.each do |inputs|
        pp neuralnet.get_forward_outputs(inputs)[1]
      end
  end

end
