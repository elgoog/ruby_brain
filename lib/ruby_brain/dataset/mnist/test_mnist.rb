require 'mnist'


train_images = Mnist.load_images('./train-images-idx3-ubyte.gz')
train_labels = Mnist.load_labels('./train-labels-idx1-ubyte.gz')

puts train_images[0].class
puts train_images[1].class
puts train_images[2].size
puts train_images[2][0].size
puts train_images[2][59999][783].class
puts train_images[2][59999].class
puts "------------------------------"

10.times do |j|
  train_images[2][j].unpack('C*').each_with_index do |e, i|
    print(e > 50 ? 'x' : ' ')
    puts if (i % 28) == 0
  end
  puts
  puts train_labels[j]
end






