# WCEClassifyViStA-
WCE Classification with Visual Understanding through Segmentation and Attention

Our WCE image classification network is named as “**WCE ClassifyViStA (WCE Classification with Visual understanding through Segmentation and Attention)**”. A block schematic of ClassifyViStAis shown below. It has a standard classification path and two other branches viz. implicit attention branch and the segmentation branch. 

![](Images/framework.png){width="6.260416666666667in"
height="3.4791666666666665in"}

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

**Table 3: Pictures of any 10 best images selected from the validation
dataset showing its classification and detection**

While there are many images from the validation set with high confidence
and high IoU detection, we have chosen 10 images varying in
illumination, region (inside the body), texture etc. and that which
covers bounding boxes with small, large and medium areas that overlap
significantly with the corresponding groundtruth. The groundtruth
detection is also shown in the table below for easy comparison.

*Table 4: Pictures of the achieved interpretability plot of any 10 best
images selected from the validation dataset**

Our team intends to explain the classification using the predicted
segmentation mask. In fact, a doctor would classify a WCE image as
bleeding by looking for bleeding spots in the image. So, enabling the
machine to attempt to mimic what doctor does is, we believe, the more
natural way of explaining the reason for classification. Towards this
end, we had added a parallel branch to the classifier network which
decodes the features from the classifier backbone (in the style of U-net
decoder) and predicts a segmentation mask. Below, we show the predicted
segmentation masks where the brighter corresponds to area where possible
bleeding is present in the original WCE image. The corresponding
groundtruth image with the inlaid bounding box is shown for easy
comparison. For sake of consistency, we are choosing the same 10 images
that was chosen for the above table (Table 3), even though there are
many other images for which the segmentation mask is more accurate
(segmentation mask for all the 263 positive class validation images is
provided in the results folder here). Even for the chosen 10 images,
except for a couple of images, the masks that explain the classification
are quite accurate.

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  S.No.   Image Name      Groundtruth image with ground truth bbox                                                               Predicted segmentation mask for explainability
  ------- --------------- ------------------------------------------------------------------------------------------------------ ------------------------------------------------------------------------------------------------------
  1       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (276).png       (276).png](Images/image2.png){width="1.5833333333333333in"    (276).png](Images/image22.png){width="1.582638888888889in"
                          height="1.5833333333333333in"}                                                                         height="1.582638888888889in"}

  2       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (320).png       (320).png](Images/image4.png){width="1.582638888888889in"     (320).png](Images/image23.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  3       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (473).png       (473).png](Images/image6.png){width="1.582638888888889in"     (473).png](Images/image24.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  4       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (581).png       (581).png](Images/image8.png){width="1.582638888888889in"     (581).png](Images/image25.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  5       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (654).png       (654).png](Images/image10.png){width="1.582638888888889in"    (654).png](Images/image26.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  6       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (697).png       (697).png](Images/image12.png){width="1.582638888888889in"    (697).png](Images/image27.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  7       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (775).png       (775).png](Images/image14.png){width="1.582638888888889in"    (775).png](Images/image28.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  8       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (894).png       (894).png](Images/image16.png){width="1.582638888888889in"    (894).png](Images/image29.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  9       bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (968).png       (968).png](Images/image18.png){width="1.582638888888889in"    (968).png](Images/image30.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}

  10      bleeding/img-   ![C:\\Users\\bala\\Downloads\\outputs\\img-                                                            ![C:\\Users\\bala\\Downloads\\inverted_seg_inferences\\img-
          (1120).png      (1120).png](Images/image20.png){width="1.582638888888889in"   (1120).png](Images/image31.png){width="1.582638888888889in"
                          height="1.582638888888889in"}                                                                          height="1.582638888888889in"}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



