# Neural-Style-Transfer
A C++ implementation for Neural-Style-Transfer


Based on [Adrian](https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/)'s example, here you are the C++ version.
The original neural style transfer algorithm was introduced by Gatys et al. in their 2015 paper, [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

## Dependencies
In order to make it as simple as possible, the major pipeline between Adrian's and my implemenation is identical. So what you need is
1. [OpenCV with DNN module](https://github.com/opencv/opencv).

An there is some minor difference in the implementation after the inference step. 

C++
```c++
cv::dnn::imagesFromBlob(outBlob,results);
```

Python
```python
output = output.reshape((3, output.shape[2], output.shape[3]))
```

The rest of the pipeline is the same.
