require 'pathname'

module RubyBrain
  class TrainingDataManipulator
    attr_accessor :columns
    def initialize(data_file, has_header)
      puts data_file
      @columns = []
      @raw_data = parse_data(data_file, has_header)
    end

    def parse_data(data_file, has_header)
      array_of_data_set = []
      File.open(data_file) do |f|
        @columns = f.readline.chomp.split(',') if has_header
        f.each_line do |line|
          next if /\A\s+\z/ =~ line
          array_of_data_set << line.chomp.split(',')
        end
      end
      array_of_data_set
    end

    def ix(*col_index)
      @raw_data.map do |a_set|
        a_set.values_at(*col_index).map(&:to_f)
      end
    end

    def num_data_sets
      @raw_data.length
    end

  end
end
