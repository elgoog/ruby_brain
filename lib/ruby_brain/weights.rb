module RubyBrain
  class WeightContainer
    def initialize(num_units_list)
      @w_3d = []

      num_units_list.each_cons(2) do |num_units_on_left_layer, num_units_on_right_layer|
        @w = []
        (num_units_on_left_layer + 1).times do |i|
          @w[i] = []
          num_units_on_right_layer.times do |j|
            @w[i][j] = Random.rand(2.0) - 1.0
          end
        end
        @w_3d << @w
      end
    end

    def load_from(weights_set_source)
      @w_3d = weights_set_source
    end

    def get_weights_as_array
      @w_3d.dup
    end

    def overwrite_weights(weights_set_source)
      # weights_set_source.each_with_index do |weights, i|
      #   next if i >= @w_3d.size
      #   weights.each_with_index do |w, j|
      #     next if j >= @w_3d[i].size
      #     w.size.times do |k|
      #       @w_3d[i][j][k] = w[k] unless w[k].nil?
      #     end
      #   end
      # end

      @w_3d.zip(weights_set_source).each_with_index do |wl, i|
        next if wl[1].nil?
        wl[0].zip(wl[1]).each_with_index do |ww, j|
          next if ww[1].nil?
          ww[0].zip(ww[1]).each_with_index do |w, k|
            @w_3d[i][j][k] = w[1] || w[0]
          end
        end

      end
    end

    def num_sets
      @w_3d.size
    end

    def weights_of_order(order_number)
      @w_3d[order_number]
    end

    def dump_to_yaml(file_name=nil)
      if file_name
        File.open file_name, 'w+' do |f|
          YAML.dump(@w_3d, f)
        end
      end
      # # @w_3d.to_yaml
    end

    def load_from_yaml_file(yaml_file)
      overwrite_weights(YAML.load_file(yaml_file))
    end

    def each_weights
      @w_3d.each do |weights|
        yield weights
      end
    end

    def each_weights_with_index
      @w_3d.each_with_index do |weights, i|
        yield weights, i
      end
    end
  end
end
