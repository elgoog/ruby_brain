# RubyBrain

RubyBrain is a library of neural net, deep learning for Ruby.
You can install/use this library easily because the core is created by using only Ruby standard library.

The code of RubyBrain is the neuron oriented style.
This means that a class which represents a neuraon exists and each neurons are instances of the class.
So, you can treat neurons flexibly in a network.
Instead, the speed is very slow and it might not be reasonable for applications to use this library in the core.
However this library may help you get more deep knowledge around neuralnet/deep learning.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'ruby_brain'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install ruby_brain

## Usage

Please refer to
[github.com/elgoog/ruby_brain/README.org](https://github.com/elgoog/ruby_brain/README.org)
for detail.

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/elgoog/ruby_brain.


## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

