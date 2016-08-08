require "ruby_brain/version"
require 'yaml'
require 'pp'

module RubyBrain
  require "ruby_brain/nodes"
  require "ruby_brain/layer"
  require "ruby_brain/weights"
  require "ruby_brain/network"
  require "ruby_brain/trainer"
  require 'ruby_brain/exception'
  require 'ruby_brain/training_data_manipulator'
  
  module Nodes end
  module Trainer end
  module Exception end
  module DataSet end
end
