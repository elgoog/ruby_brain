module RubyBrain
  class Network
    attr_accessor :learning_rate
    
    # Constructor of Network class
    # 
    # @example network structure Array(num_units_list)
    #  [10, 30, 3]       # => 3 inputs, hidden layer 1 with 30 units, 3 outputs
    #  [15, 50, 60, 10]  # => 15 inputs, hidden layer 1 with 50 units, hidden layer 2 with 60 units, 10 outputs
    # 
    # @param num_units_list [Array] Array which describes the network structure
    def initialize(num_units_list)
      @layers = []
      @num_units_list = num_units_list
      @weights_set = WeightContainer.new(@num_units_list)
    end

    def load_weights_from(weights_set_source)
      @weights_set.load_from(weights_set_source)
      init_network
    end

    # def overwrite_weights(weights_set_source)
    #   @weights_set.overwrite_weights(weights_set_source)
    # end

    # Initialize the network. This method creates network actually based on the network structure Array which specified with Constructor.
    def init_network
      @layers = []
      layer = Layer.new
      (@num_units_list[0] + 1).times do
        layer.append Nodes::ConstNode.new
        layer.output_weights = @weights_set.weights_of_order(0)
      end
      @layers << layer

      @num_units_list[1..-2].each_with_index do |num_units, i|
        layer = Layer.new
        layer.input_weights = @weights_set.weights_of_order(i)
        layer.output_weights = @weights_set.weights_of_order(i+1)
        (num_units).times do
          layer.append Nodes::Neuron.new
        end
        layer.append Nodes::ConstNode.new
        @layers << layer
      end

      layer = Layer.new
      layer.input_weights = @weights_set.weights_of_order(@num_units_list.size - 2)
      @num_units_list[-1].times do
        layer.append Nodes::Neuron.new
      end
      @layers << layer
    end

    # def get_weights_as_array
    #   @weights_set.get_weights_as_array
    # end

    # Calculate the network output of forward propagation.
    #
    # @param inputs [Array] Input dataset.
    def get_forward_outputs(inputs)
      inputs.each_with_index do |input, i|
        @layers.first.nodes[i].value = input
      end

      a_layer_outputs = nil
      a_layer_inputs = @layers.first.forward_outputs
      @layers.each do |layer|
        a_layer_outputs = layer.forward_outputs(a_layer_inputs)
        a_layer_inputs = a_layer_outputs
      end
      a_layer_outputs
    end
    
    # Calculate the networkoutput of backward propagation.
    #
    # @param backward_inputs [Array] Input for backpropagation. Usually it is loss values.
    def run_backpropagate(backward_inputs)
      a_layer_outputs = nil
      a_layer_inputs = backward_inputs
      @layers.reverse[0..-2].each do |layer|
        a_layer_outputs = layer.backward_outputs(a_layer_inputs)
        a_layer_inputs = a_layer_outputs
      end
      a_layer_outputs
    end

    # Updates weights actually based on the result of backward propagation
    #
    def update_weights
      @weights_set.each_weights_with_index do |weights, i|
        weights.each_with_index do |wl, j|
          wl.each_with_index do |w, k|
            wl[k] = w - (@learning_rate * @layers[i].nodes[j].this_output * @layers[i+1].nodes[k].this_backward_output)
          end
        end
      end
    end

    def update_weights_of_layer(layer_index)
      layer_index = @weights_set.num_sets + layer_index if layer_index < 0
      @weights_set.each_weights_with_index do |weights, i|
        next if i != layer_index
        weights.each_with_index do |wl, j|
          wl.each_with_index do |w, k|
            wl[k] = w - (@learning_rate * @layers[i].nodes[j].this_output * @layers[i+1].nodes[k].this_backward_output)
          end
        end
      end
    end

    # def calculate_rms_error(training_inputs_set, training_outputs_set)
    #   accumulated_errors = 0.0
    #   training_inputs_set.zip(training_outputs_set).each do |t_input, t_output|
    #     forward_outputs = get_forward_outputs(t_input)
    #     total_error_of_output_nodes = 0.0
    #     forward_outputs.zip(t_output).each do |o, t|
    #       total_error_of_output_nodes += (o - t)**2 / 2.0
    #     end
    #     accumulated_errors += total_error_of_output_nodes / forward_outputs.size
    #   end
    #   Math.sqrt(2.0 * accumulated_errors / training_inputs_set.size)
    # end
    
    # Starts training with training dataset
    #
    # @param inputs_set [Array<Array>] Input dataset for training. The structure is 2 dimentional Array. Eatch dimentions correspond to samples and features.
    # @param outputs_set [Array<Array>] Output dataset for training. The structure is 2 dimentional Array. Eatch dimentions correspond to samples and features.
    # @param max_training_count [Integer] Max training count.
    # @param tolerance [Float] The Threshold to stop training. Training is stopped when RMS error reach to this value even if training count is not max_training_count.
    # @param monitoring_channels [Array<Symbol>] Specify which log should be reported. Now you can select only `:best_params_training`
    def learn(inputs_set, outputs_set, max_training_count=50, tolerance=0.0, monitoring_channels=[])
      raise RubyBrain::Exception::TrainingDataError if inputs_set.size != outputs_set.size
      #      raise "inputs_set and outputs_set has different size!!!!" if inputs_set.size != outputs_set.size

      best_error = Float::INFINITY
      best_weights_array = []
      max_training_count.times do |i_training|
        accumulated_errors = 0.0 # for rms
        inputs_set.zip(outputs_set).each do |t_input, t_output|
          forward_outputs = get_forward_outputs(t_input)
          # for rms start
          total_error_of_output_nodes = forward_outputs.zip(t_output).reduce(0.0) do |a, output_pair|
            a + ((output_pair[0] - output_pair[1])**2 / 2.0)
          end
          # end
          accumulated_errors += total_error_of_output_nodes / forward_outputs.size
          # accumulated_errors += forward_outputs.zip(t_output).reduce(0.0) { |a, output_pair| a + ((output_pair[0] - output_pair[1])**2 / 2.0) } / forward_outputs.size
          # for rms end
          backward_inputs = forward_outputs.zip(t_output).map { |o, t| o - t }
          run_backpropagate(backward_inputs)
          update_weights
        end

        rms_error = Math.sqrt(2.0 * accumulated_errors / inputs_set.size) # for rms
        # rms_error = calculate_rms_error(inputs_set, outputs_set)
        puts "--> #{rms_error} (#{i_training}/#{max_training_count})"
        
        if rms_error < best_error
          puts "update best!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
          best_error = rms_error
          best_weights_array = @weights_set.get_weights_as_array
        end
        puts "best: #{best_error}"


        break if rms_error <= tolerance
      end
      if monitoring_channels.include? :best_params_training
        File.open "best_weights_#{Time.now.to_i}.yml", 'w+' do |f|
          YAML.dump(best_weights_array, f)
        end
      end
    end


    def learn2(inputs_set, outputs_set, max_training_count=50, tolerance=0.0, monitoring_channels=[])
      # looks like works well for networks which has many layers... [1, 10, 10, 10, 1], [1, 100, 100, 100, 1]
      # looks like NOT works well for networks which has many units in a layer... [1, 100, 1]
      raise RubyBrain::Exception::TrainingDataError if inputs_set.size != outputs_set.size
      # raise "inputs_set and outputs_set has different size!!!!" if inputs_set.size != outputs_set.size
      initial_learning_rate = @learning_rate

      rms_error = Float::INFINITY
      max_training_count.times do |i_training|
        accumulated_errors = 0.0 # for rms
        inputs_set.zip(outputs_set).each do |t_input, t_output|
          forward_outputs = get_forward_outputs(t_input)
          # for rms start
          total_error_of_output_nodes = forward_outputs.zip(t_output).reduce(0.0) do |a, output_pair|
            a + ((output_pair[0] - output_pair[1])**2 / 2.0)
          end
          # end
          error_of_this_training_data = total_error_of_output_nodes / forward_outputs.size
          accumulated_errors += error_of_this_training_data
          # accumulated_errors += forward_outputs.zip(t_output).reduce(0.0) { |a, output_pair| a + ((output_pair[0] - output_pair[1])**2 / 2.0) } / forward_outputs.size
          # for rms end
          # if error_of_this_training_data > rms_error**2/2.0
          #   @learning_rate *= 10.0
          # end
          backward_inputs = forward_outputs.zip(t_output).map { |o, t| o - t }
          run_backpropagate(backward_inputs)
          update_weights
          # @learning_rate = initial_learning_rate
        end

        rms_error = Math.sqrt(2.0 * accumulated_errors / inputs_set.size) # for rms

        # rms_error = calculate_rms_error(inputs_set, outputs_set)
        puts "--> #{rms_error} (#{i_training}/#{max_training_count})"
        break if rms_error <= tolerance
      end
    end


    def learn_only_specified_layer(layer_index, inputs_set, outputs_set, max_training_count=50, tolerance=0.0)
      # looks like works well for networks which has many layers... [1, 10, 10, 10, 1], [1, 100, 100, 100, 1]
      # looks like NOT works well for networks which has many units in a layer... [1, 100, 1]
      raise "inputs_set and outputs_set has different size!!!!" if inputs_set.size != outputs_set.size
      initial_learning_rate = @learning_rate

      rms_error = Float::INFINITY
      max_training_count.times do |i_training|
        accumulated_errors = 0.0 # for rms
        inputs_set.zip(outputs_set).each do |t_input, t_output|
          forward_outputs = get_forward_outputs(t_input)
          # for rms start
          total_error_of_output_nodes = forward_outputs.zip(t_output).reduce(0.0) do |a, output_pair|
            a + ((output_pair[0] - output_pair[1])**2 / 2.0)
          end
          # end
          error_of_this_training_data = total_error_of_output_nodes / forward_outputs.size
          accumulated_errors += error_of_this_training_data
          # accumulated_errors += forward_outputs.zip(t_output).reduce(0.0) { |a, output_pair| a + ((output_pair[0] - output_pair[1])**2 / 2.0) } / forward_outputs.size
          # for rms end
          if error_of_this_training_data > rms_error**2/2.0
            @learning_rate *= 10.0
          end
          backward_inputs = forward_outputs.zip(t_output).map { |o, t| o - t }
          run_backpropagate(backward_inputs)
          update_weights_of_layer(layer_index)
          @learning_rate = initial_learning_rate
        end

        rms_error = Math.sqrt(2.0 * accumulated_errors / inputs_set.size) # for rms

        # rms_error = calculate_rms_error(inputs_set, outputs_set)
        puts "--> #{rms_error} (#{i_training}/#{max_training_count})"
        break if rms_error <= tolerance
      end
    end


    def dump_weights
      @weights_set.each_weights do |weights|
        pp weights
      end
    end

    # Dumps weights of the network into a file whose format is YAML.
    # 
    # @param file_name [String] The path to the YAML file in which weights are saved.
    def dump_weights_to_yaml(file_name=nil)
      @weights_set.dump_to_yaml(file_name)
    end
    
    # Loads weights of the network from existing weights file whose format is YAML.
    # 
    # @param yaml_file [String] The path to the YAML file which includes weights.
    def load_weights_from_yaml_file(yaml_file)
      @weights_set.load_from_yaml_file(yaml_file)
    end

  end
end
