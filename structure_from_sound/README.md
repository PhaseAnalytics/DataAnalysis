## Structure from Sound

This was an implementation of Sebastian Thrun's [Affine Structure from Sound](http://robots.stanford.edu/papers/thrun-structure-from-sound05.html). 

I found his description of the optimization lacking, so hopefully this notebook will help. 

In this example we simulate recording sounds at four sensors from sources located at hours on a clock, as shown in the ground truth image below. The arrows point in the direction of the sources. 

![Ground Truth][gtruth]

The estimate gives the direction of the sources relative to a 'reference' sensor, so the sensors, sources, and arrows are rotated. However, after inspection one will observe that the direction estimates are correct. 

![Source Estimates][sestimates]

[gtruth]: https://github.com/PhaseAnalytics/DataAnalysis/blob/master/structure_from_sound/sfs_groundtruth.png
[sestimates]: https://github.com/PhaseAnalytics/DataAnalysis/blob/master/structure_from_sound/sfs_estimate.png
