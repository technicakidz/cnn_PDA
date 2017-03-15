from PIL import Image
import os
from fast_layers import max_pool_forward_fast
import datetime


def jpeg2matrix(path):

	im = Image.open(path)
	m = np.asarray(im.convert('RGB')) #np.asarray:引数に配列コピーせずそのまま返す
	m = np.transpose(m, (2,0,1)).astype("float") #np.transpose(a,axes=None):転置，反転する軸を引数で指定することも可
	
	return m



def getsubject(filename):

	s = filename.split("_")
	# get subject number:
	num = int(s[3])

	return num



def max_pool_one_image(x, pool_param):
  """
  Reduce the size of an given image x using the max pool method.

  Inputs:
  - x: image matrix of shape (C,H,W) (color channels, height, width)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions
  """
  out = None
  
  # Get data dimemsions
  C, H, W = x.shape

  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']


  # Number of possible positions in height
  Hp = (H - pool_height)/stride + 1

  # Number of possible positions in width
  Wp = (W - pool_width)/stride + 1
	  
  # Initialization of output
  out = np.zeros((C, Hp, Wp))

  for c in xrange(C): # for each filter
	for i in xrange(Hp): # for each vertical possible position
	   ix = i*stride
	   for j in xrange(Wp): # for each horizontal possible position
	      jx = j*stride

	      window = x[c, ix:ix+pool_height, jx:jx+pool_width]
	      out[c,i,j] =  np.amax(window)

  return out



# This functiom is not used to obtain the current results, I used get_data_sametaum.
def get_data(nsubj=6, type="gallery"):
	"""
	Read all images for the first 'nsubj' subjects and create training or validation dataset.
	The image size is reduced from 3x256x256 to 3x128x128.

	Inputs:
	- nsubj: Number of subjects
	- type: 'gallery' or 'probes'

	Output:
	- X of shape (N,C,H,W) with N=number of images, C=channels, H=image height, W=image width
	- y of shape (N)

	"""


	if type=="probe":
		path = "/home/deepstation/workspace/2312016030/Data/PDA/prove"
	else:
		path = "/home/deepstation/workspace/2312016030/Data/PDA/gallery"

	
	# get files names
	files = os.listdir(path)

	# sort by subject number
	files.sort(key = lambda x: x.split("_")[3])

	# get subjects for each file
	sub_id = [x.split("_")[3] for x in files]
	sub_id = np.asarray(map(float, sub_id))

	# keep only the number of subjects needed
	lim = np.argmax(sub_id>nsubj) #np.argmax:引数に与えられた配列の最大インデックスを取得
	files = files[:lim]


	nfiles = len(files)

	print("\n"+type+":%d" % nfiles)

	X = np.zeros((nfiles, 3, 256, 256))　#np.zeros:要素がすべて0である0行列を新しく生成する関数
	y = np.zeros(nfiles)

	filename = files[0]

	for n in xrange(nfiles):
		if n%500==0:
			print(n)
		X[n,:,:,:] = jpeg2matrix(path+"/"+files[n])
		y[n] = getsubject(files[n])

	# Reduce size of images
	pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
	X, c = max_pool_forward_fast(X,pool_param)
	
	
	# Normalize X
	X -= np.mean(X, axis=0)　#np.mean:配列の平均

	# Cast y to int
	y = y.astype(int)

	return X, y





def get_data_sametaum(nsubj=6, type="gallery", tau=1, m=1):
	"""
	Read images created with given m and tau parameters for the first 'nsubj' 
	subjects, and create training or validation dataset.
	The image size is reduced from 3x256x256 to 3x128x128.

	Inputs:
	- nsubj: Number of subjects
	- type: 'gallery' or 'probes'
	- tau: tau recurrence plot parameter
	- m: m recurrence plot parameter 

	Output:
	- X of shape (N,C,H,W) with N=number of images, C=channels, H=image height, W=image width
	- y of shape (N)

	"""

	
	if type=="probe":
		path = "/home/deepstation/workspace/2312016030/Data/PDA/prove"
	else:
		path = "/home/deepstation/workspace/2312016030/Data/PDA/gallery"

	
	# get files names
	files = os.listdir(path)

	# keep only wanted tau and m
	taum = "m=%d_tau=%d" % (m,tau)
	files = filter(lambda x: taum in x, files)


	# sort by subject number
	files.sort(key = lambda x: x.split("_")[3])
	# get subjects for each file
	sub_id = [x.split("_")[3] for x in files]
	sub_id = np.asarray(map(float, sub_id)) #numpy

	# keep only the number of subjects needed
	lim = np.argmax(sub_id>nsubj) #numpy
	files = files[:lim]
	#print(files)

	nfiles = len(files)

	print("\n"+type+":%d" % nfiles)

	# Initialize outputs
	X = np.zeros((nfiles, 3, 256, 256)) #numpy
	y = np.zeros(nfiles) #numpy

	filename = files[0]

	for n in xrange(nfiles):
		if n%500==0:
			print(n)
		X[n,:,:,:] = jpeg2matrix(path+"/"+files[n])
		y[n] = getsubject(files[n])

	# Reduce size of images
	pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
	X, c = max_pool_forward_fast(X,pool_param)
	
	
	# Normalize X
	X -= np.mean(X, axis=0) #numpy

	# Cast y to int
	y = y.astype(int)

	return X, y




# This function is not used
def get_data_sametaum_nopool(nsubj=6, type="gallery", tau=1, m=1):
	"""
	Read images created with given m and tau parameters for the first 'nsubj' 
	subjects, and create training or validation dataset.
	The image size is NOT REDUCED.

	Inputs:
	- nsubj: Number of subjects
	- type: 'gallery' or 'probes'
	- tau: tau recurrence plot parameter
	- m: m recurrence plot parameter 

	Output:
	- X of shape (N,C,H,W) with N=number of images, C=channels, H=image height, W=image width
	- y of shape (N)

	"""

	
	if type=="probe":
		path = "/home/deepstation/workspace/2312016030/Data/PDA/prove"
	else:
		path = "/home/deepstation/workspace/2312016030/Data/PDA/gallery"

	
	# get files names
	files = os.listdir(path)

	# keep only wanted tau and m
	taum = "m=%d_tau=%d" % (m,tau)
	files = filter(lambda x: taum in x, files)


	# sort by subject number
	files.sort(key = lambda x: x.split("_")[3])
	# get subjects for each file
	sub_id = [x.split("_")[3] for x in files]
	sub_id = np.asarray(map(float, sub_id)) #numpy

	# keep only the number of subjects needed
	lim = np.argmax(sub_id>nsubj) #numpy
	files = files[:lim]
	#print(files)

	nfiles = len(files)

	print("\n"+type+":%d" % nfiles)

	X = np.zeros((nfiles, 3, 256, 256)) #numpy
	y = np.zeros(nfiles)#numpy

	filename = files[0]

	for n in xrange(nfiles):
		if n%500==0:
			print(n)
		X[n,:,:,:] = jpeg2matrix(path+"/"+files[n])
		y[n] = getsubject(files[n])

	# Reduce size of images
	#pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
	#X, c = max_pool_forward_fast(X,pool_param)
	
	
	# Normalize X
	X -= np.mean(X, axis=0)#numpy

	# Cast y to int
	y = y.astype(int)

	return X, y




def write_report(nsubj, nf, fs, hd, ws, reg, lr, nep, bs, trainacc, valacc, loss, time, comm=" "):
	"""
	Write a report at the end of each training to keep track of the different parameters and results.
	"""


	s= datetime.datetime.now().strftime("%I%M%p_%B_%d")

	file = open("/home/gakusei/Documents/log/"+s+".txt", "w")

	file.write("LOSS: %f \t" % loss)
	file.write("TRAINING ACC: %f \t" % trainacc)
	file.write("VAL ACC: %f \n\n" % valacc)
	file.write("time: %f \n\n" % time)

	file.write("Classes: %d \n" % nsubj)
	file.write("Number of filters: %d \n" % nf)
	file.write("Filter size: %d \n" % fs)
	file.write("Hidden dims: %d \n" % hd)


	file.write("Weight regularization: %f \n" % ws)
	file.write("learning rate: %f \n" % lr)
	file.write("Number of epochs: %d \n" % nep)
	file.write("Batch size: %d \n" % bs)

	file.write("Comment: %s" % comm)



	file.close()

	return 0
