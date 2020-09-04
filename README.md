# FaceNav

An experiment in image based face tracking to support
neurosurgical navigation.

This code experiments with the idea of using a
Azure Kinect device (circa 2020) to localize
a human face and spatially register it to a
CT or MRI scan coordinate system.

The code is posted here for reference and further study.  There is [sample data](https://github.com/pieper/facenav/releases/download/v0.1/MRHead-face.zip) in the asset section of the first release.  The MRHead is a scan of me from 2002, when I was about 50 pounds heavier, which explains why the cheeks don't match.

## Built with

* [3D Slicer](https://slicer.org)
* [Azure Kinect DK (and device)](https://docs.microsoft.com/en-us/azure/kinect-dk/)
* [Intel open3d](https://github.com/intel-isl/Open3D)
* [TensorFlow](https://www.tensorflow.org/)
* [mtcnn](https://pypi.org/project/mtcnn/)

## Examples

Click the images to view video versions on youtube.

[![MRI tracking face in 3D](http://img.youtube.com/vi/T5218dK9CH0/0.jpg)](http://www.youtube.com/watch?v=T5218dK9CH0 "Live face navigation")

[![MRI tracked face in 3D](http://img.youtube.com/vi/7-rj6pQ_A6c/0.jpg)](http://www.youtube.com/watch?v=7-rj6pQ_A6c "Example registration")
