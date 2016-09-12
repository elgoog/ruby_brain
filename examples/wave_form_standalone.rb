require "ruby_brain"

X = (0..1).step(0.01).to_a
Y_IDEAL = X.map {|x| (0.75 * Math.sin(x*2*Math::PI) - 0.2 * Math.cos(5*x*2*Math::PI - 0.023) + 1) / 2} 
Y = [Y_IDEAL, Array.new(X.size) {rand(-0.05..0.05)}].transpose.map {|e| e.inject(:+)}
a_network = RubyBrain::Network.new([1, 13, 6, 1])
a_network.init_network
a_network.learning_rate = 0.5
a_network.learn(X.map{|e| [e]}, Y.map{|e| [e]}, max_tra2ining_count=40000, tolerance=0.0004, monitoring_channels=[:best_params_training])
Y_PREDICATED = X.map{|e| [e]}.map {|a| a_network.get_forward_outputs(a).first}

