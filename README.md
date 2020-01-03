# ImagePatcher

Use MASK-RCNN and https://github.com/JiahuiYu/generative_inpainting to remove humans from images.

### Some examples:

#### Input:
<img src="samples/small_regensburg.jpg" width ="400">

#### First Detect People using Mask R-CNN (we used boxes instead of masks as this works better for inpainting)
<img src="samples/input_small_regensburg.png" width ="400">

#### Then Run Generative Inpainting to fill the missing parts
<img src="samples/output_small_regensburg.png" width ="400">


#### Or with a different picture
<img src="samples/small_topview.jpg" width ="400">
<img src="samples/input_small_topview.png" width ="400">
<img src="samples/output_small_topview.png" width ="400">

