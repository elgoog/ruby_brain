

module RubyBrain
  module Exception

    class RubyBrainError < StandardError
    end


    class DataDimensionError < RubyBrainError
    end

    class TrainingDataError < DataDimensionError
    end

  end
end
