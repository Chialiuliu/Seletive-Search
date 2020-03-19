import os
import cv2
import numpy as np

def get_file_name(dir):
	return os.listdir(dir)

def graph_based_image_segmentation(img, sigma, k_cluster, min_area):
	"""
		Use Graph-Based Image Segmentation to get initial images

		and concate these regions into axis = 2
	"""
	graphBasedSegment = cv2.ximgproc.segmentation.createGraphSegmentation(
											sigma = sigma, k = k_cluster, min_size = min_area)
	graphSegment = graphBasedSegment.processImage(img)
	graphSegment = graphSegment[:, :, np.newaxis]
	img = np.concatenate((img, graphSegment), axis = 2)

	return img

def region_calcu(img):
	"""
		Calculate color histogram and texture histogram between regions

		Args:
			img (numpy.array): img infos by pixel

		Returns:
			dictionary, bounding box, color histogram, texture histogram and size.

		Example :
			[0] => 	["bbox"],
					["size"],
					["hist_color"],
					["hist_texture]
	"""
	'''
		Step 1 :: get graph-based image segmentation bounding box
	'''
	regions = {}
	for i in range(np.max(img[:, :, -1])):
		y, x = np.where(img[:, :, -1] == i)
		leftX, upY, rightX, downY = min(x), min(y), max(x), max(y)
		regions[i] = {"bbox": [leftX, upY, rightX, downY, i]}

	'''
		Step 2 :: get histogram of each region
	'''
	imageLBP = LBP(img[:, :, :-1])
	for regionNumber, data in list(regions.items()):
		maskRegion = img[:, :, :-1][img[:, :, -1] == regionNumber]

		regions[regionNumber]["hist_color"] = calcu_hist(maskRegion, BINS = 25, bound = (0,255))
		regions[regionNumber]["hist_texture"] = calcu_hist(
				imageLBP[img[:, :, -1] == regionNumber],
				BINS = 10,
				bound = (0, 255)
			)
		regions[regionNumber]["size"] = len(maskRegion) / 3

	return regions

def LBP(img):
	'''
		Implement Local Binary Pattern algorithm

		Example : 
			(decimal)		LBP				(binary)				return			(decimal)
			200, 100, 0				00000100, 00000011, 00000111				 4,   3,   7
			255,  50, 6		=>		00000000, 11000111, 10000111		=>		 0, 199, 135
			125,  50, 8				01000000, 11000001, 10000001				64, 193, 129

			LBP = (binary) 11000111 = (decimal) 199
	'''
	height, width, dimension = img.shape
	result = np.zeros([height, width, dimension])

	move_x = [-1, 0, 1, 1, 1, 0, -1, -1]
	move_y = [-1, -1, -1, 0, 1, 1, 1, 0]

	for channel in range(3):
		eachChannel = img[:, :, channel]

		for y in range(0, height):
			for x in range(0, width):
				binary = []

				for i in range(len(move_x)):
					try:
						if ((eachChannel[y + move_y[i]][x + move_x[i]]) >= (eachChannel[y][x])):
							binary.append(1)
						else:
							binary.append(0)
					except:
						binary.append(0)

					# caculate binary to decimal
					p = 0
					for k in range(len(binary)):
						p = p + binary[k] * (2 ** k)
					result[y][x][channel] = p
	
	"""
	# using skimage Library
	import skimage.feature
	for channel in range(3):
		result[:, :, colour_channel] = skimage.feature.local_binary_pattern(img[:, :, channel], 8, 1.0)
	#cv2.imwrite("wow.jpg", result[:,:,0])
	"""

	return result

def calcu_hist(region, BINS, bound = (0, 255), channelNum = 3):
	"""
		if in color histogram

		each channel get 25 bins

		for a RGB 3-channels image, the histogram of return will be 25 * 3 = 75 bins,

		if in texture

		each channel get 10 bins.
	"""
	hist = np.array([])
	
	for channel in range(channelNum):
		eachChannel = region[:, channel]
		
		hist = np.concatenate([hist, np.histogram(eachChannel, BINS, bound)[0]])

	hist /= len(region)

	return hist

def neighbor_calcu(regions):
	"""
		Calculate the intersection pairs of regions

		Args:
			regions (list): region infos

		Returns:
			array: neighbor pairs
	"""
	regions = list(regions.items())
	
	neighborHood = []
	for regionA, dataA in regions[: -1]:
		for regionB, dataB in regions[regionA + 1 : ]:
			if is_neighbor(dataA, dataB):
				neighborHood.append((regionA, regionB))

	return neighborHood

def is_neighbor(A, B):
	"""
		Return if two regions are intersection

		Args:
			A (dictionary): primary region
			B (dictionary): other region

		Returns:
			bool
	"""
	xL = max(A['bbox'][0], B['bbox'][0])
	yU = max(A['bbox'][1], B['bbox'][1])
	xR = min(A['bbox'][2], B['bbox'][2])
	yD = min(A['bbox'][3], B['bbox'][3])

	if (xR - xL > 0) or (yD - yU > 0):
		return True

	return False

def similarity_calcu(A, B, imageArea):
    """
		Calculate the sum of similarities in color, texture, size and fill

		Args:
			A (list): primary region info
			B (list): other region
			imageArea (int): the amount of pixels

		Returns:
			int, similarity
	"""
	return (color_similar(A, B) + texture_similar(A, B) +
			size_similar(A, B, imageArea) + fill_similar(A, B, imageArea))

def color_similar(A, B):
	"""
		Get the sum of minimum one in each color histogram interval between region A and B.

		Args:
			A (list): primary region info
			B (list): other region

		Returns:
			float, color similarity.
	"""
	return (sum(np.minimum(A["hist_color"], B["hist_color"])))

def texture_similar(A, B):
	"""
		Get the sum of minimum one in each texture histogram interval between region A and B.

		Args:
			A (list): primary region info
			B (list): other region

		Returns:
			float, texture similarity.
	"""
	return (sum(np.minimum(A["hist_texture"], B["hist_texture"])))

def size_similar(A, B, imageArea):
    """
		Get size similarity between A and B.

		Args:
			A (list): primary region info
			B (list): other region
			imageArea (int): amount of original image

		Returns:
			float, size similarity.
	"""
	return 1.0 - ((A['size'] + B['size']) / imageArea)

def fill_similar(A, B, imageArea):
	"""
		Get fill similarity between A and B.
		mergeBBoxArea is the bounding box of A after merge B

		it is size(BB) in the paper

		Args:
			A (list): primary region info
			B (list): other region
			imageArea (int): amount of original image

		Returns:
			float, fill similarity.
	"""
	bBoxWidth = max(A['bbox'][2], B['bbox'][2]) - min(A['bbox'][0], B['bbox'][0])
	bBoxHeight = max(A['bbox'][3], B['bbox'][3]) - min(A['bbox'][1], B['bbox'][1])
	mergeBBoxArea = bBoxWidth * bBoxHeight

	return 1.0 - ((mergeBBoxArea - A['size'] - B['size']) / imageArea)

def region_merge(A, B, sequence):
    """
		Merge two regions and re-calculate their bounding box, color histogram, texture histogram and size.

		Args:
			A (list): region
			B (list): region
			sequence (int): the order in all of the regions

		Returns:
			dictionary, bounding box, color histogram, texture histogram and size.
	"""
	mergeSize = A['size'] + B['size']
	mergeRegion = {
		'bbox': [min(A['bbox'][0], B['bbox'][0]), min(A['bbox'][1], B['bbox'][1]),
				max(A['bbox'][2], B['bbox'][2]), max(A['bbox'][3], B['bbox'][3]), sequence],
		'hist_color': (A['hist_color'] * A['size'] + B['hist_color'] * B['size']) / mergeSize,
		'hist_texture': (A['hist_texture']  *A['size'] + B['hist_texture'] * B['size']) / mergeSize,
		'size': mergeSize
	}

	return mergeRegion

def selective_search(img, sigma= 0.01, k_cluster= 300, min_area= 5000):
	'''
		Step 1 :: graph-based image segmentation
	'''
	imgSegment = graph_based_image_segmentation(img, sigma= sigma, k_cluster= k_cluster, min_area= min_area)

	'''
		Step 2 :: calculate region and histogram
	'''
	region = region_calcu(imgSegment)

	'''
		Step 3 :: calculate neighbor regions
	'''
	neighborHood = neighbor_calcu(region)

	'''
		Step 4 :: calculate initialized similarity in all of the neighbor regions.
	'''
	regionSimilar = {}
	imgArea = img.shape[0] * img.shape[1]
	for regionA, regionB in neighborHood:
		regionSimilar[(regionA, regionB)] = similarity_calcu(
				region[regionA],
				region[regionB],
				imgArea
			)

	'''
		Step 5 :: select the region pair that has highest similarity and merge them until array is empty
	'''
	while regionSimilar != {}:
    	'''
			find and merge the highest similarity region pair
		'''
		regionA, regionB = sorted(regionSimilar.items(), key= lambda i: i[1])[-1][0]

		# 1. add new region
		newRegionPos = len(region.items())
		region[newRegionPos] = region_merge(region[regionA], region[regionB], newRegionPos)

		# 2. delete regions have been merged
		similarDelete = []
		for regionPair, data in list(regionSimilar.items()):
			if (regionA in regionPair) or (regionB in regionPair):
				similarDelete.append(regionPair)
				del regionSimilar[regionPair]
		# 3. calculate similarity of new region with neighbors
		for pair in [regionWithoutAB for regionWithoutAB in similarDelete if regionWithoutAB != (regionA, regionB)]:
			otherRegion = pair[0] if pair[1] in (regionA, regionB) else pair[1]

			regionSimilar[(newRegionPos, otherRegion)] = similarity_calcu(region[newRegionPos],
				region[otherRegion], imgArea)

	'''
		Step 6 :: package region infos for return
	'''
	resultRegion = {}
	for regionPos, data in list(region.items()):
		resultRegion[regionPos] = {
				'regionXL': data['bbox'][0],
				'regionYU': data['bbox'][1],
				'regionXR': data['bbox'][2],
				'regionYD': data['bbox'][3],
				'regionSize': data['size'],
				'regionOrder': data['bbox'][4]
		}

	return resultRegion

def rect_draw(img, region, color = (0, 0, 255), regionSize = 2000):
	for num, rect in list(region.items()):
		if rect['regionSize'] < regionSize:
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
		img = cv2.imread(filePath + name)
		region = selective_search(img, 0.9, 500, 10)
		img = rect_draw(img, region, color = (0, 0, 255), regionSize = 2000)
		cv2.imwrite(savePath + 'SS_' + name, img)
		print("TIME : {}".format(time.time() - t))
