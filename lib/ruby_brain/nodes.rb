module RubyBrain::Nodes
  class Neuron
    attr_accessor :order_index, :left_side_weights, :right_side_weights
    attr_reader :this_output, :this_backward_output

    def initialize(gain=1.0)
      @gain = gain
      @this_output = nil
      @this_backward_output = nil
    end

    def get_sigmoid_output(sigmoid_input)
      1.0 / (1 + Math.exp(-1 * @gain * sigmoid_input))
    end

    def get_backward_sigmoid_output(backward_sigmoid_input)
      @gain * (1 - @this_output) * @this_output * backward_sigmoid_input
    end

    def output_of_forward_calc(inputs)
      sigmoid_input = 0.0
      @left_side_weights.transpose[@order_index].zip(inputs).each do |weight, input|
        sigmoid_input += input * weight
      end
      @this_output = get_sigmoid_output(sigmoid_input)
    end

    def output_of_backward_calc(backward_inputs)
      sigmoid_backward_input = 0.0
      if @right_side_weights.nil?
        sigmoid_backward_input = backward_inputs[@order_index]
      else
        @right_side_weights[@order_index].zip(backward_inputs).each do |weight, input|
          sigmoid_backward_input += input * weight
        end
      end
      @this_backward_output = get_backward_sigmoid_output(sigmoid_backward_input)
    end
  end

  class ConstNode
    attr_accessor :order_index, :value, :left_side_weights, :right_side_weights
    attr_reader :this_output, :this_backward_output

    def initialize(value=1.0)
      @value = value
      @this_output = nil
    end

    def output_of_forward_calc(inputs=[])
      @this_output = @value
    end

    def output_of_backward_calc(backward_inputs=[])
      nil
    end
  end
end  
