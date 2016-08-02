module RubyBrain::DataSet::Mnist

  require 'mnist'
  require 'open-uri'
  
  def download_file(target_url, dest_path)
    File.open(dest_path, "wb") do |saved_file|
      open(target_url, "rb") do |read_file|
        saved_file.write(read_file.read)
      end
    end
  end

  def data
    train_images_path = Dir.pwd + '/train-images-idx3-ubyte.gz'
    train_labels_path = Dir.pwd + '/train-labels-idx1-ubyte.gz'
    test_images_path = Dir.pwd + '/t10k-images-idx3-ubyte.gz'
    test_labels_path = Dir.pwd + '/t10k-labels-idx1-ubyte.gz'

    unless File.exist?(train_images_path)
      puts 'downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ...'
      download_file('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', train_images_path) 
    end
    
    unless File.exist?(train_labels_path)
      puts 'downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ...'
      download_file('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', train_labels_path) 
    end

    unless File.exist?(test_images_path)
      puts 'downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz ...'
      download_file('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', test_images_path) 
    end
    
    unless File.exist?(test_labels_path)
      puts 'downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz ...'
      download_file('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', test_labels_path) 
    end

    train_images = Mnist.load_images(train_images_path)
    train_labels = Mnist.load_labels(train_labels_path)
    test_images = Mnist.load_images(test_images_path)
    test_labels = Mnist.load_labels(test_labels_path)

    input_training_set = train_images[2].map do |image|
      image.unpack('C*').map {|e| e / 255.0}
    end
    
    output_training_set = train_labels.map do |label|
      one_hot_vector = Array.new(10, 0)
      one_hot_vector[label] = 1
      one_hot_vector
    end

    input_test_set = test_images[2].map do |image|
      image.unpack('C*').map {|e| e / 255.0}
    end
    
    output_test_set = test_labels.map do |label|
      one_hot_vector = Array.new(10, 0)
      one_hot_vector[label] = 1
      one_hot_vector
    end

    # puts train_images[0].class
    # puts train_images[1].class
    # puts train_images[2].size
    # puts train_images[2][0].size
    # puts train_images[2][59999][783].class
    # puts train_images[2][59999].class
    # puts "------------------------------"

    # 10.times do |j|
    #   train_images[2][j].unpack('C*').each_with_index do |e, i|
    #     print(e > 50 ? 'x' : ' ')
    #     puts if (i % 28) == 0
    #   end
    #   puts
    #   puts train_labels[j]
    # end
    
    [{input: input_training_set, output: output_training_set}, {input: input_test_set, output: output_test_set}]
  end
  
  module_function :data, :download_file
end
