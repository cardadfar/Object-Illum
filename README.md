# Object-Illum

Object-Illum is a novel relighting technique that relights a sequence of images based on global and local scales.

<img src='../master/results/sample-results.png' style='width: 50%;'>

You can access the [paper](http://cardadfar.com/assets/project-descriptions/images/object-illum/paper.pdf) here.

## Algorithm

### Dependencies

* PIL
* os
* time
* numpy
* scipy
* matplotlib
* colorsys
* sklearn
* glob
* cv2

### Contributions

* ```yolo_opencv.py``` was based around the following [repo](https://github.com/arunponnusamy/object-detection-opencv).
* ```relight.py``` hue chart visualization was based around the following [repo](https://github.com/lighttransport/colorcorrectionmatrix).
* ```relight.py``` global relight algorithm was based around the following [paper](https://ieeexplore.ieee.org/document/8300518/).


### Running the Algorithm

At the end of ```relight.py```, you can specify the following parameters,
```
input_dir = "inputs/"
output_dir = "outputs/"
file_types = [".jpg"]
iters = 1
num_hues = 10
num_lums = 1
num_sats = 1
hls_steps = [1.0, 1.0, 1.0]

main(input_dir, output_dir, iters, num_hues, num_lums, num_sats, hls_steps)
```

* input_dir: directory to import files from
* output_dir: directory to save relighting files from
* file_types: types of files to accept
* iters: number of relighting iterations to run
* num_hues: number of hue bins to create on startup
* num_lums: number of luminance bins to create on startup
* num_sats: number of saturation bins to create on startup
* hls_steps: timestep for [hue, luminance, saturation] relighting masks

To run the code, execute the following:
```
python3 relight.py
```

### Debugging Features

The main ```relight.py``` file contains additional features to help debug color correction and relighting. Toggling them will perform the operations commented below.

```
# returns relight mask of hue, luminance, and saturation
show_plots = False

# returns segmentation plot of global hue classes
show_class = False

# returns segmentation plot of global hue class averages
show_seg = False

# returns comparison chart of global hues before and after relighting
show_charts = False

# saves image to output directory
save = True
```
