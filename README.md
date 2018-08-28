# Seletive-Search
- [more detail] (http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)
# Setup
- Windows 10
- Python 3.4

# Result
<img src="./result/result.jpg" width="700px/">

# Detail

1. Selective Search uses graph-based image segmentation to get initial regions, first.  
   In my practice, I just use the function of graph-based image segmentation in openCV.

2. Calculate similarities between each neighboring region.  
   A. **_color similarity_**:
       In order to measure the similarity in color, this approach can measure distribution
       of each region.
       calculate color histogram of each region, than calculate the sum of minimize
       similarity in each interval between both neighboring regions.

<img src="./algorithm/color.jpg" width="500px/">  

       We can easily get the color similarity after merge two region by using following
       algorithm.

<img src="./algorithm/merge.jpg" width="500px/">  

   B. **_texture similarity_**:
       In this paper, they use SIFT to get image textures.
       In my practice, I use LBP instead.

<img src="./algorithm/texture.jpg" width="500px/">  

   C. **_size similarity_**:

<img src="./algorithm/size.jpg" width="500px/">  

   D. **_fill similarity_**:

<img src="./algorithm/fill.jpg" width="500px/">  
3.

```
