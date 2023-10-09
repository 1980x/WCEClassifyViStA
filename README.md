# WCEClassifyViStA-
WCE Classification with Visual Understanding through Segmentation and Attention

Our WCE image classification network is named as “**WCE ClassifyViStA (WCE Classification with Visual understanding through Segmentation and Attention)**”. A block schematic of ClassifyViStAis shown below. It has a standard classification path and two other branches viz. implicit attention branch and the segmentation branch. 

![](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.001.png)

In the implicit attention branch, the feature maps from the encoder are weighted by the given groundtruth segmentation masks and then sent to the classification head. This ensures that, for bleeding class, the network focuses on the region of bleeding rather than the whole image. 

In the segmentation branch, the output from the encoder goes through a U-net style decoder and constructs the segmentation mask. 

Both these branches supplement the standard path of classification implicitly with the region of interest in case of bleeding images. For non-bleeding images, the attention branch focuses on the whole feature map while the segmentation branch constructs a zero-filled segmentation map. 

While inferencing, attention branch is not used since segmentation masks will not be available during inference. Class predictions are obtained using the standard classification path and explanations are derived from the predicted segmentation masks. Like doctors look for bleeding spots to classify a WCE image into bleeding class, the predicted segmentation masks identify bleeding regions with bright pixels and non-bleeding regions with dark pixels, attempting to mimic what the doctor does. This is the most natural way of explaining the class prediction for a WCE image. So, we have come with our own explainabilty for explaining the class predictions instead of relying on LIME, SHAP etc. 

To improve classification performance, we have used an ensemble of two tow models in our Classify ViStA viz. Resnet18 and VGG16. The final classification is based average prediction probabilities from both the models.

**Detection Network (SoftNMS activated YOLOV8)**

For WCE bleeding region detection, we have used YOLOV8. However, instead of using YOLOV8 as it, we activated soft non-max suppression during during inferencing (Soft NMS) instead of standard NMS which provides a softer and more nuanced handling of overlapping boxes, reducing the risk of removing partially correct boxes. We found soft NMS to do better than standard NMS in our validation set. 

**A set of tables of the achieved evaluation metrics (for 80:20 train:valid split)**

Train-valid split was obtained using *sklearn.model\_selection.train\_test\_split* with *random\_state* argument set to 42 for reproducibility.

Table 1: Classification Performance on Validation Set

|**S.No.**|**Metric**|**Value**|
| :- | :- | :- |
|1|Accuracy|0\.9962|
|2|Precision|0\.9962|
|3|Recall|0\.9962|
|4|F1-Score|0\.9962|

Table 2: Detection Performance on Validation Set

|**S.No.**|**Metric**|**Value**|
| :- | :- | :- |
|1|Average Precision|0\.7715|
|2|MAP@0.5|0\.726|
|3|MAP@0.5-0.95|0\.483|
|4|Average IoU|0\.6405|

**Table 3: Pictures of any 10 best images selected from the validation dataset showing its classification and detection**

While there are many images from the validation set with high confidence and high IoU detection, we have chosen 10 images varying in illumination, region (inside the body), texture etc. and that which covers bounding boxes with small, large and medium areas that overlap significantly with the corresponding groundtruth. The groundtruth detection is also shown in the table below for easy comparison.

|S.No.|Image Name|Groundtruth image with ground truth bbox|Predicted bbox with confidence|Classification + Confidence|
| :- | :- | :- | :- | :- |
|1|bleeding/img- (276).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (276).png]</p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (276).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.003.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 1.0</p>|
|2|bleeding/img- (320).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (320).png]</p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (320).png](Aspose.Words.c8aee428-Images/0d98-4d6a-b174-1bab7a6ef67e.005.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 1.0</p>|
|3|bleeding/img- (473).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (473).png]</p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (473).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.007.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 1.0</p>|
|4|bleeding/img- (581).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (581).png]</p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (581).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.009.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 0.9998</p>|
|5|bleeding/img- (654).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (654).png]</p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (654).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.011.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 0.9998</p>|
|6|bleeding/img- (697).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (697).png]</p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (697).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.013.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 0.9994</p>|
|7|bleeding/img- (775).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (775).png]</p><p></p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (775).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.015.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 0.9982</p>|
|8|bleeding/img- (894).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (894).png]</p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (894).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.017.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 0.9997</p>|
|9|bleeding/img- (968).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (968).png]</p><p></p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (968).png](Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.019.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 0.9994</p>|
|10|bleeding/img- (1120).png|<p></p><p>![C:\Users\bala\Downloads\outputs\img- (1120).png]</p>|<p></p><p>![C:\Users\bala\Downloads\images\img- (1120).png](Images/Aspose.Words.c8aee428-0d98-4d6a-b174-1bab7a6ef67e.021.png)</p><p></p>|<p>Predicted: Bleeding</p><p>Confidence: 1.0</p>|

