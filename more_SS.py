import os
import cv2
import numpy as np

def get_file_name(dir):
	return os.listdir(dir)

def graph_based_image_segmentation(img, sigma, k_cluster, min_area):
	graphBasedSegment = cv2.ximgproc.segmentation.createGraphSegmentation(
											sigma= sigma, k= k_cluster, min_size= min_area)
	graphSegment = graphBasedSegment.processImage(img)
	graphSegment = graphSegment[:,:,np.newaxis]
	img = np.concatenate((img, graphSegment), axis= 2)

	return img

def region_calcu(img):
	#Step 1- get graph-based image segmentation bounding box
	REGION = {}
	for i in range(np.max(img[:,:,-1])):
		y, x = np.where(img[:,:,-1]== i)
		leftX, upY, rightX, downY = min(x), min(y), max(x), max(y)
		REGION[i] = {"bbox":[leftX, upY, rightX, downY, i]}
		
	#Step 2- get each region's histogram
	# Local Binary Parameter
	imageLBP = LBP(img[:,:,:-1])
	#print("IMAGELBP SHAPE: {}".format(imageLBP.shape))
	for regionNumber,data in list(REGION.items()):
		#print(regionNumber)
		
		maskRegion = img[:,:,:-1][img[:,:,-1]== regionNumber]
		#print("LBP:{}".format(imageLBP))
		#print(maskRegion)
		#print(len(maskRegion))
		REGION[regionNumber]["hist_color"] = calcu_hist(maskRegion, BINS=25, bound=(0,255))
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		REGION[regionNumber]["hist_texture"] = calcu_hist(imageLBP[img[:,:,-1]== regionNumber],
																		 BINS=10, bound=(0,255))
		REGION[regionNumber]["size"] = len(maskRegion)/3
	return REGION

def LBP(img):
	
	h, w, d = img.shape
	result = np.zeros([h,w,d])
	move_x = [-1, 0, 1, 1, 1, 0, -1, -1]
	move_y = [-1, -1, -1, 0, 1, 1, 1, 0]
	
	# follow LBP algorithm self
	for channel in range(3):
		eachChannel = img[:,:,channel]
		for y in range(0,h):
			for x in range(0,w):
				binary = []
				for i in range(len(move_x)):
					try:
						if ((eachChannel[y+move_y[i]][x+move_x[i]]) >= (eachChannel[y][x])):
							binary.append(1)
						else:
							binary.append(0)
					except:
						binary.append(0)
					p=0
					for k in range(len(binary)):
						p = p+ binary[k]*(2**k)
					result[y][x][channel] = p
	#cv2.imwrite("wow.jpg",result[:,:,0])
	
	"""
	# using skimage Library
	import skimage.feature
	for channel in range(3):
		result[:, :, colour_channel] = skimage.feature.local_binary_pattern(img[:, :, channel], 8, 1.0)
	#cv2.imwrite("wow.jpg", result[:,:,0])
	"""
	return result

def calcu_hist(region, BINS, bound=(0,255)):
	"""
		if in color hist

		each channel get 25 bins

		for a RGB 3channels  image, the return histogram will be 25*3 = 75 bins
	"""
	hist = np.array([])
	
	for channel in range(3):
		eachChannel = region[:,channel]
		
		hist = np.concatenate([hist,np.histogram(eachChannel, BINS, bound)[0]])

	hist /= len(region)

	return hist

def neighbor_calcu(R):
	region = list(R.items())
	neighborHood = []

	for regionA,dataA in region[:-1]:
		for regionB,dataB in region[regionA+ 1: ]:
			if is_neighbor(dataA, dataB):
				neighborHood.append((regionA, regionB))
	return neighborHood

def is_neighbor(A, B):
	xL = max(A['bbox'][0],B['bbox'][0])
	yU = max(A['bbox'][1],B['bbox'][1])
	xR = min(A['bbox'][2],B['bbox'][2])
	yD = min(A['bbox'][3],B['bbox'][3])

	if (xR-xL<= 0) or (yD-yU<= 0):
		return False
	return True

def similarity_calcu(A, B, imageArea):
	return (color_similar(A, B)+ texture_similar(A, B)+
			size_similar(A, B, imageArea)+ fill_similar(A, B, imageArea))

def color_similar(A, B):
	return (sum(np.minimum(A["hist_color"],B["hist_color"])))

def texture_similar(A, B):
	return (sum(np.minimum(A["hist_texture"],B["hist_texture"])))

def size_similar(A, B, imageArea):
	return 1.0- ((A['size']+ B['size'])/ imageArea)

def fill_similar(A, B, imageArea):
	"""
		mergeBBoxArea is the bounding box of A merge B

		it is the bbsize in the paper
	"""

	w = max(A['bbox'][2], B['bbox'][2])- min(A['bbox'][0], B['bbox'][0])
	h = max(A['bbox'][3], B['bbox'][3])- min(A['bbox'][1], B['bbox'][1])
	mergeBBoxArea = w*h
	return 1.0- ((mergeBBoxArea- A['size']- B['size'])/ imageArea)

def region_merge(A, B, pos):
	mergeSize = A['size']+ B['size']
	mergeRegion = {
		'bbox': [min(A['bbox'][0], B['bbox'][0]), min(A['bbox'][1], B['bbox'][1]),
				max(A['bbox'][2], B['bbox'][2]), max(A['bbox'][3], B['bbox'][3]), pos],
		'hist_color': (A['hist_color']*A['size']+ B['hist_color']*B['size'])/ mergeSize,
		'hist_texture': (A['hist_texture']*A['size']+ B['hist_texture']*B['size'])/ mergeSize,
		'size': mergeSize
	}
	return mergeRegion

def selective_search(img, sigma= 0.01, k_cluster= 300, min_area= 5000):
	# Step 1- graph-based image segmentation
	imgSegment = graph_based_image_segmentation(img, sigma= sigma, k_cluster= k_cluster, min_area= min_area)
	# Step 2- calculate region and histogram
	region = region_calcu(imgSegment)
	# Step 3- calculate neighbors
	neighborHood = neighbor_calcu(region)
	# Step 4- calculate initial similarity
	regionSimilar = {}
	imgArea = img.shape[0]*img.shape[1]
	#print('REGION:{}'.format(list(region.items())[1][0]))
	for regionA, regionB in neighborHood:
		regionSimilar[(regionA, regionB)] = similarity_calcu(region[regionA],
																	region[regionB], imgArea)

	# Step 5- select the highest similarity regions and merge the 2 regions, until array is empty
	while regionSimilar!= {}:
		# find and merge the highest similarity of two regions
		regionA, regionB = sorted(regionSimilar.items(),key= lambda i: i[1])[-1][0]

		# 1. add new region
		newRegionPos = len(region.items())
		#print(regionA, regionB)
		region[newRegionPos] = region_merge(region[regionA], region[regionB], newRegionPos)
		# 2. delete those regions have been merged
		similarDelete = []
		for regionPos, data in list(regionSimilar.items()):
			if (regionA in regionPos) or (regionB in regionPos):
				similarDelete.append(regionPos)
				del regionSimilar[regionPos]
		# 3. calculate similarity of new region with neighbors
		for pair in [regionWithoutAB for regionWithoutAB in similarDelete if regionWithoutAB!= (regionA, regionB)]:
			otherRegion = pair[0] if pair[1] in (regionA, regionB) else pair[1]
			regionSimilar[(newRegionPos, otherRegion)] = similarity_calcu(region[newRegionPos],
																				region[otherRegion], imgArea)
	# Step 6- sort out region
	resultRegion = {}
	for regionPos, data in list(region.items()):
		resultRegion[regionPos] = {
				'regionXL': data['bbox'][0],
				'regionYU': data['bbox'][1],
				'regionXR': data['bbox'][2],
				'regionYD': data['bbox'][3],
				'regionSize': data['size'],
				'regionNum': data['bbox'][4]
		}
	return resultRegion

def rect_draw(img, region, color=(0, 0, 255), regionSize= 2000):
	for num, rect in list(region.items()):
		if rect['regionSize']< regionSize:
			continue 
		cv2.rectangle(img, (rect['regionXL'], rect['regionYU']), (rect['regionXR'], rect['regionYD']), color, 1)
	return img

if __name__ == '__main__':
	import time
	t = time.time()
	filePath = './images/'
	savePath = ''
	fileName = get_file_name(filePath)
	for name in fileName:
		img = cv2.imread(filePath+ name)
		region = selective_search(img, 0.9, 500, 10)
		img = rect_draw(img, region, color= (0, 0, 255), regionSize= 2000)
		cv2.imwrite(savePath+ 'SS_'+ name, img)
		print("TIME : {}".format(time.time()-t))
