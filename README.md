# Object-Illum

Object-Illum is a novel relighting technique that relights a sequence of images based on global and local scales.

<img src='../master/results/sample-results.png' style='width: 50%;'>

You can access the [paper](http://cardadfar.com/assets/project-descriptions/images/light-stabilization/paper.pdf) here.

### Running the Algorithm


At the end relight.py, you can specify the following parameters,
```
input_dir = "test-outputs/gates-low-res/"
output_dir = "greyscale/"
iters = 1
num_hues = 10
num_lums = 1
num_sats = 1
hls_steps = [1.0, 1.0, 1.0]

main(input_dir, output_dir, iters, num_hues, num_lums, num_sats, hls_steps)
```

