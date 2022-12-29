# classification-patch-base-system

## Introduction
Machine learning has been very successfull for many years. It could have successfully solved many challanges such as detection of tumors from brain image [1], segmentation of problematic area [2], and classification of certain image [3]. However, supervised learning methodology has a fundamental limitation due to its dependancy to the given label [4]. Since the learning is based on the given label, it is even more serious especially when the label is ambiguous [5]. However, it is hard to blame the label itself because, in many cases, it is very difficult to provide clear label due to the feature of given image data. Here are the challenge I prepared:

## Challenge

![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/original_img.png) \
I have a wooden fence with full of climbing plants in my house. \
I want to quantitate the coverage of the vegetation. \
So I made label like below:

![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/lable_img.png) \
However, the label (green mask) includes the wall area as well which is not desirable because the wall area is obviously non-vegetation. If the wall area is included in the training, the model will learn the false information and the consequence will be not welcomed. But it also takes considerably extensive amount of time and labour to annotate only vegetation avoiding all the wall area. Therefore, it is generally very hard to expect to get 100% decent label data with finely carved every single edge. This happens a lot especially when you collaborate with other data providers. So I believe this must be overcome by the power of algorithm. To break this challange, here I implemented ambiguous data-resistant patch-based classification:

## Overview of the algorithm
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/classify_example.gif)

## Another challenge
To provide training dataset for classification algorithm, you must decide vegetation and non-vegetation(wall) from cropped images. Let's see some of the cropped images. \
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/type1.png) \
This picture looks obvious. \
These cropped patches are defintely **'vegetation'**. \
\
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/background.png) \
Also, these examples are obviously in the **'wall'** group. \
\
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/type2.png) \
What about these? \
Eventhough it looks bit 'less vegie', it is still agreeable to be **'vegetation'** group \
\
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/type3.png) \
Then what about these? \
Will you put these in the **'vegetation'** group? or **'wall'** group? \
Here is the dilemma. If you include these to 'vegetation' group, the model will learn **'wall'** pixels data as well.\
However, if you include these to **'wall'** group, the opposite thing will happen. \
\
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/type4.png) \
Now, What about these? \ 
Will you put these patches to which group? **Vegetation?** or **Wall?** \
\
As you can see. It's hard to decide...
![picutre](https://github.com/boguss1225/classification-patch-base/blob/main/screenshot/meme.png)

So, I will let these be decided by algorithm by voting from inside of the patches!
To lower the work overload, this vote will happens only when the classification confidence of the patch is lower than a certain thresh hold

## Voting system 
(TO BE DESCRIBED)

## Test Result
* Data overview

## Conclusion
This would be useful when you want...
* general coverage of certain type of object
* distinguish background

## References
