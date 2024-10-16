<div align="center"> Official repository of </div>

# <div align="center"> GraFIQs: Face Image Quality Assessment <br> Using Gradient Magnitudes </div>

<div align="center", padding="30px">
  <span style="display:block; height: 20em;">&emsp;</span>
  <p><b>Jan Niklas Kolf</b><sup> 1,2</sup> &emsp; <b>Naser Damer</b><sup> 1,2</sup> &emsp; <b>Fadi Boutros</b><sup> 1</sup></p>
  <p><sup>1 </sup>Fraunhofer IGD &emsp; <sup>2 </sup>Technische Universität Darmstadt</p>
  <p>Accepted at CVPR Workshop 2024</p>
</div>

<div align="center">
        <a href="https://arxiv.org/pdf/2404.12203.pdf"><img src="https://github.com/jankolf/assets/blob/main/badges/download-paper-arxiv-c1.svg?raw=true"></a>
        &emsp;
        <a href="https://share.jankolf.de/s/WWCXmNkj7FTcRpR"><img src="https://github.com/jankolf/assets/blob/main/badges/download-models-c1.svg?raw=true"></a>
</div>


## <div align="center"> Overview 🔎 </div>
<div align="center">
    <img width="50%" src="https://raw.githubusercontent.com/jankolf/assets/main/GraFIQs/overview.svg">
</div>

An overview of the proposed GraFIQs for assessing the quality of unseen testing samples.
Sample *I* is passed into the pretrained face recognition model and Batch Normalization statistics are extracted. Then, the mean-squared-error between the Batch Normalization statistics obtained by processing the testing sample and the one recorded during the face recognition training is calculated.
The mean-squared-error is backpropagated through the pretrained face recognition model to extract the gradient magnitudes of parameter group φ.
Finally, the absolute sum of gradient magnitudes of φ is calculated and utilized as Face Image Quality.


## <div align="center"> Abstract 🤏 </div>

Face Image Quality Assessment (FIQA) estimates the utility of face images for automated face recognition (FR) systems. We propose in this work a novel approach to assess the quality of face images based on inspecting the required changes in the pre-trained FR model weights to minimize differences between testing samples and the distribution of the FR training dataset. To achieve that, we propose quantifying the discrepancy in Batch Normalization statistics (BNS), including mean and variance, between those recorded during FR training and those obtained by processing testing samples through the pretrained FR model. We then generate gradient magnitudes of pretrained FR weights by backpropagating the BNS through the pretrained model. The cumulative absolute sum of these gradient magnitudes serves as the FIQ for our approach. Through comprehensive experimentation, we demonstrate the effectiveness of our training-free and quality labeling-free approach, achieving competitive performance to recent state-of-theart FIQA approaches without relying on quality labeling, the need to train regression networks, specialized architectures, or designing and optimizing specific loss functions.

## <div align="center"> Usage 🖥 </div>

### Setup

Install all necessary packages in a Python >=3.10 environment:
```
   pip install torch torchvision numpy tqdm opencv-python
```

### Extract Face Image Quality Scores

To extract scores for images in a folder,
1. download pre-trained model weights [from this link](https://share.jankolf.de/s/WWCXmNkj7FTcRpR) and place them in a location of your choice
2. run `python extract_grafiqs.py` and set arguments accordingly
    ```
    usage: extract_grafiqs.py [-h] [--image-path IMAGE_PATH] [--image-extension IMAGE_EXTENSION]
                          [--output-dir OUTPUT_DIR] [--backbone {iresnet50,iresnet100}]
                          [--weights WEIGHTS] [--gpu GPU]
                          [--path-replace PATH_REPLACE] [--path-replace-with PATH_REPLACE_WITH]
                          [--bgr2rgb]
    GraFIQs

    options:
    -h, --help            show this help message and exit
    --image-path IMAGE_PATH
                            Path to images.
    --image-extension IMAGE_EXTENSION
                            Extension/File type of images (e.g. jpg, png).
    --output-dir OUTPUT_DIR
                            Directory to write score files to (will be created if it does not exist).
    --backbone {iresnet50,iresnet100}
                            Backbone architecture to use.
    --weights WEIGHTS     Path to backbone architecture weights.
    --gpu GPU             GPU to use.
    --path-replace PATH_REPLACE
                            Prefix of image path which shall be replaced.
    --path-replace-with PATH_REPLACE_WITH
                            String that replaces prefix given in --path-replace.
    --bgr2rgb             If specified, changes color space of CV2 image from BGR (default) to RGB.
    ```

### Evaluation and EDC curves

Please refer to [CR-FIQA repository](https://github.com/fdbtrs/CR-FIQA/tree/main) for evaluation and EDC plotting.

## <div align="center"> Citation ✒ </div>

If you found this work helpful for your research, please cite the article with the following bibtex entry:
```
@inproceedings{DBLP:conf/cvpr/KolfDB22,
  author       = {Jan Niklas Kolf and
                  Naser Damer and
                  Fadi Boutros},
  title        = {GraFIQs: Face Image Quality Assessment Using Gradient Magnitudes},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2024 - Workshops, Seattle, WA, USA, June 17-18, 2024},
  pages        = {1490--1499},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/CVPRW63382.2024.00156},
  doi          = {10.1109/CVPRW63382.2024.00156},
  timestamp    = {Thu, 10 Oct 2024 17:01:03 +0200},
  biburl       = {https://dblp.org/rec/conf/cvpr/KolfDB22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## License ##

This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

Copyright (c) 2024 Fraunhofer Institute for Computer Graphics Research IGD, Darmstadt.
