#Import system for comman line processing
import sys

#Image processing import
import skimage
from skimage import feature
from skimage import io
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

#Handle video inputs
#Note, need to make sure that imageio-ffmpeg has been pip installed
import imageio

#Plotting
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

#Numerical analysis
import numpy as np

#Optimization
import scipy as scipy
import scipy.optimize as opt

#File handling
import sys

function, image, *kwargs = sys.argv

if sys.platform == 'linux' and 'home' in image and image[0] != '/':
	image = f'/{image}'

#Set default numerical arguments
baselineThreshold = 20
linThreshold = 20
circleThreshold = 5
everyNSeconds = 1
baseOrder = 1
σ = 1
ε = 1e-2
startSeconds = 10
tolerance = 8
fitType = 'linear'
lim = 10

#Get the file type for the image file
parts = image.split('.')
ext = parts[-1]

video = False

if ext.lower() == 'avi' or ext.lower() == 'mp4':
	video = True
elif ext.lower() != 'jpg' and ext.lower() != 'png' and ext.lower() != 'gif':
	raise ValueError(f'Invalid file extension provided. I can\'t read {ext} files')

#Overwrite these defaults if desired
kwargs = zip ( * [ iter(kwargs) ] * 2 )

for argPair in kwargs:
	if argPair[0] == '-b' or argPair[0] == '--baselineThreshold':
		baselineThreshold = int(argPair[1])
	elif argPair[0] == '-o' or argPair[0] == '--baselineOrder':
		baseOrder = int(argPair[1])
	elif argPair[0] == '-c' or argPair[0] == '--circleThreshold':
		circleThreshold = int(argPair[1])
	elif argPair[0] == '-l' or argPair[0] == '--linThreshold':
		linThreshold = int(argPair[1])
	elif (argPair[0] == '-t' or argPair[0] == '--times') and video:
		everyNSeconds = float(argPair[1])
	elif argPair[0] == '-s':
		σ = float(argPair[1])
	elif (argPair[0] == '-ss' or argPair[0] == '--startSeconds') and video:
		startSeconds = float(argPair[1])
	elif (argPair[0] == '--fitType'):
		fitType = argPair[1].replace(" ","").lower()


# Only if we are using a linear fit should we define the dist function
if fitType == 'circular':
	# Define the loss function that we use for fitting
	def dist(param, points):
		*z , r = param
		ar = [(np.linalg.norm(np.array(z) - np.array(point)) - r ) ** 2 
				for point in points]
		return np.sum(ar)

### Strategy for automating contact angle measurements for Lauren+McLain

### Use some toolkit to extract every nth frame from avi file

### At every extracted frame, read it as a grayscale numpy array

if not video:
	images = [io.imread(image,as_gray = True)]
else:
	images = imageio.get_reader(image)
	fps = images.get_meta_data()['fps']

	#Set the conversion from RGB to grayscale using the scikit-image method (RGB to grayscale page)
	conversion = np.array([0.2125,0.7154,0.0721])

	#Perform the conversion and extract only so many frames to analyze
	images = [ np.dot(im,conversion) for i,im in enumerate(images) 
				if ( ( i / np.round(fps) - startSeconds ) % everyNSeconds ) == 0 and
				   ( i / np.round(fps) > startSeconds ) ]
	images = [im / im.max() for im in images]

# Get 4-list of points for left, right, top, and bottom crop (in that order)

# Show the first image in the stack so that user can select crop box
print('Waiting for your input, please crop the image as desired and hit enter')
viewer = ImageViewer(images[0])
rect_tool = RectangleTool(viewer, on_enter = viewer.closeEvent)
viewer.show()
cropPoints = np.array(rect_tool.extents)
cropPoints = np.array(np.round(cropPoints),dtype = int)

time = []
angles = []
volumes = []
baselineWidth = []

# Make sure that the edges are being detected well
edges = feature.canny(images[0],sigma = σ)
fig , ax = plt.subplots(2,1,gridspec_kw = {'height_ratios': [10,1]} , figsize = (8,8))
ax[0].imshow(edges, cmap = 'gray_r', vmin = 0, vmax = 1)
ax[0].set_xlim(cropPoints[:2])
ax[0].set_ylim(cropPoints[2:])
ax[0].axis('off')

sigmaSlide = widgets.Slider(ax[1],r'$\log_{10}\sigma$',-1,1,valinit = np.log10(σ),color = 'gray')

def update(val):
	edges = feature.canny(images[0], sigma = np.power(10,val))
	ax[0].imshow(edges,cmap = 'gray_r',vmin = 0,vmax = 1)
	fig.canvas.draw_idle()

sigmaSlide.on_changed(update)
print('Waiting for your input, please select a desired filter value, and close image when done')
plt.show()
σ = np.power(10,sigmaSlide.val)
print(f'Proceeding with sigma = {σ : 6.2f}')

# Create a set of axes to hold the scatter points for all frames in the videos
plt.figure()
scatAx = plt.axes()

plt.figure(figsize = (5,5))
imAxes = plt.axes()
plt.ion()
plt.show()

for j,im in enumerate(images):
	### Using scikit-image canny edge detection, find the image edges
	edges = feature.canny(im,sigma = σ)

	# Obtain the X,Y coordinates of the True values in this edge image 
	# (for processing)
	coords = np.array([[i,j] for j,row in enumerate(edges) 
							 for i,x in enumerate(row) if x])

	

	# Crop the set of points that are going to be used for analysis
	crop = np.array([[x,y] for x,y in coords if (x >= cropPoints[0] and
												 x <= cropPoints[1] and 
												 y >= cropPoints[2] and 
												 y <= cropPoints[3])])

	# Get the baseline from the left and right threshold pixels of the image 
	# (this is important not to crop too far)
	baseline = {'l':np.array([[x,y] for x,y in coords 
									if (x >= cropPoints[0] and 
										x <= cropPoints[0] + baselineThreshold and 
										y >= cropPoints[2] and 
										y <= cropPoints[3] )]),
				'r':np.array([[x,y] for x,y in coords 
									if (x >= cropPoints[1] - baselineThreshold and 
										x <= cropPoints[1] and 
										y >= cropPoints[2] and 
										y <= cropPoints[3] )])}

	# Fit the baseline to a line of form y = Σ a[i]*x**i for i = 0 .. baseOrder using np.linalg
	X = np.ones((baseline['l'].shape[0] + baseline['r'].shape[0],baseOrder + 1))
	x = np.concatenate((baseline['l'][:,0],baseline['r'][:,0]))
	for col in range(baseOrder + 1):
		X[:,col] = np.power(x,col)

	y = np.concatenate((baseline['l'][:,1],baseline['r'][:,1]))
	a = np.linalg.lstsq(X,y, rcond = None)[0]

	baseF = lambda x: np.dot(a,np.power(x,range(baseOrder + 1)))

	assert ( len(a) == baseOrder + 1 )

	# Now find the points in the circle
	circle = np.array([(x,y) for x,y in crop if
											y - (np.dot(a,np.power(x,range(baseOrder + 1))))  <= -circleThreshold])

	# Make sure that flat droplets (wetted) are ignored (i.e. assign angle to NaN and continue)
	if circle.shape[0] < 5:
		angles += [ (np.NaN, np.NaN) ]
		time += [ j * everyNSeconds ]
		baselineWidth += [ np.NaN ]
		break	

	scatAx.scatter(circle[:,0],circle[:,1])

	# Plot the current image
	imAxes.clear()
	imAxes.imshow(im,cmap = 'gray', vmin = 0, vmax = 1)
	imAxes.axis('off')

	# Baseline
	x = np.linspace( 0 , im.shape[1] )
	y = np.dot(a,np.power(x, [ [po]*len(x) for po in range(baseOrder + 1) ] ) )
	imAxes.plot(x,y,'r-')

	if fitType == 'linear':

		# Look for the greatest distance between points on the baseline
		# Future Mike here: apparently I had the idea that a drop and baseline must be the same color - bad idea! There can still be edges at the bottom of the drop.
		# To rectify this, I need some way of detecting where the circle pops up. Just look for points off of the baseline?

		# Magic 2 just to get the top three rows of the circle to find where the edges are
		offBaseline = np.array([ (x,y) for x,y in circle if y  - (np.dot(a,np.power(x,range(baseOrder + 1)))) >= -(circleThreshold + linThreshold)])

		limits = [ np.amin(offBaseline[:,0]) , np.amax(offBaseline[:,0]) ]
		
		# Get linear points
		linearPoints = {'l':np.array( [ (x,y) for x,y in crop if 
								   (y - (np.dot(a,np.power(x,range(baseOrder + 1)))) <= -circleThreshold and y - (np.dot(a,np.power(x,range(baseOrder + 1)))) >= -(circleThreshold + linThreshold)) and
								   ( x <= limits[0] + linThreshold/2 ) and ( x >= limits[0] - linThreshold/2 ) ] ),
						'r':np.array( [ (x,y) for x,y in crop if 
								   (y - (np.dot(a,np.power(x,range(baseOrder + 1)))) <= -circleThreshold and y - (np.dot(a,np.power(x,range(baseOrder + 1)))) >= -(circleThreshold + linThreshold)) and
								   ( x <= limits[1] + linThreshold/2 ) and ( x >= limits[1] - linThreshold/2 ) ] ) }

		L = np.ones( ( linearPoints['l'].shape[0] , 2 ) )
		L[:,1] = linearPoints['l'][:,0]
		l = linearPoints['l'][:,1]

		R = np.ones( ( linearPoints['r'].shape[0] , 2 ) )
		R[:,1] = linearPoints['r'][:,0]
		r = linearPoints['r'][:,1]

		params = {'l':np.linalg.lstsq(L,l,rcond=None),
				  'r':np.linalg.lstsq(R,r,rcond=None)}

		# Initialize paramater dictionaries
		b = {}
		m = {}

		# Initialize vector dictionary
		v = {}

		# Initialize vertical success dictionary
		vertical = {'l':False,'r':False}

		# Define baseline vector - slope will just be approximated from FD at each side (i.e. limit[0] and limit[1])
		bv = {'l': [1, (baseF(limits[0] + ε/2) - baseF(limits[0] - ε/2))/ε ]/np.linalg.norm([1,(baseF(limits[0] + ε/2) - baseF(limits[0] - ε/2))/ε]) ,
			  'r': [1, (baseF(limits[1] + ε/2) - baseF(limits[1] - ε/2))/ε ]/np.linalg.norm([1,(baseF(limits[1] + ε/2) - baseF(limits[1] - ε/2))/ε]) }

		# Define vectors from fitted slopes

		for side in ['l','r']:
			# Get the values for this side of the drop
			fits, residual, rank, singularValues = params[side]
			
			# Extract the parameters
			b[side], m[side] = fits

			# Do all the checks I can think of to make sure that fit succeeded
			if rank != 1 and np.prod(singularValues) > linearPoints[side].shape[0] and residual < tolerance: #Check to make sure the line isn't vertical
				v[side] = [ 1 , m[side] ] / np.linalg.norm ( [ 1 , m[side] ] )
			else:		
				#Okay, we've got a verticalish line, so swap x <--> y and fit to c' = A' * θ'
				Aprime = np.ones( ( linearPoints[side].shape[0] , 2 ) )
				Aprime[:,1] = linearPoints[side][:,1]
				cprime = linearPoints[side][:,0]

				#Now fit to a vertical-ish line (x = m'y + b')
				new_params = np.linalg.lstsq(Aprime,cprime,rcond=None)

				b[side], m[side] = new_params[0]

				v[side] = [ m[side] , 1 ] / np.linalg.norm ( [ m[side] , 1 ] )

				vertical[side] = True

		# Reorient vectors to compute physically correct angles
		if v['l'][1] > 0:
			v['l'] = -v['l']
		if v['r'][1] < 0:
			v['r'] = -v['r']

		# Calculate the angle between these two vectors defining the base-line and tangent-line
		ϕ = { i : np.arccos ( np.dot ( bv[i] , v[i] ) ) * 360 / 2 / np.pi for i in ['l','r'] }

		# Plot lines
		for side in ['l','r']:
			x = np.linspace( 0 , im.shape[1] )
			if not vertical[side]:
				y = m[side] * x + b[side]
			else:
				y = np.linspace( 0 , im.shape[0] )
				x = m[side] * y + b[side]
			imAxes.plot(x,y,'r-')

		bWidth = limits[1] - limits[0]

		V = np.NaN
		# TODO:// Add the actual volume calculation here!

	elif fitType == 'circular':
		# Get the cropped image width
		width = cropPoints[1] - cropPoints[0]

		# Try to fit a circle to the points that we have extracted, only varying the radius about the
		# center of all the points
		z = np.mean(circle, axis = 0)
		res = opt.minimize ( lambda x: dist( [*z , x] , circle ) , 
						     width/2 )

		#Get the results	
		r = res['x']
		theta = np.linspace(0,2 * np.pi,num = 500)
		x = z[0] + r * np.cos(theta)
		y = z[1] + r * np.sin(theta)

		iters = 0

		# Keep retrying the fitting while the function value is large, as this 
		# indicates that we probably have 2 circles (e.g. there's something light
		# in the middle of the image)
		while res['fun'] >= circle.shape[0] and iters < lim:
			
			# Extract and fit only those points outside the previously fit circle	
			points = np.array( [ (x,y) for x,y in circle if 
								 (x - z[0]) ** 2 + (y - z[1]) ** 2 >= r ** 2 ] )


			# Fit this new set of points, using the full set of parameters
			res = opt.minimize ( lambda x: dist( x , points ) ,
								 np.concatenate( ( np.mean( points, axis = 0) ,
								 				 [width / 4] ) ) ) 

			# Extract the new fit parameters
			*z , r = res['x']

			# Up the loop count
			#print(f"Residual of {res['fun']} at iteration {iters}")
			
			iters += 1

		# Now we need to actually get the points of intersection and the angles from
		# these fitted curves
		# Rather than brute force numerical solution, use combinations of coordinate translations and
		# rotations to arrive at a horizontal line passing through a circle

		# First step will be to translate the origin to the center-point of our fitted circle
		# x = x - z[0], y = y - z[1]
		# Circle : x**2 + y**2 = r**2
		# Line : y = m * x + (m * z[0] + b - z[1])

		# Now we need to rotate clockwise about the origin by an angle q, s.t. tan(q) = m
		# Our transformation is defined by the typical rotation matrix 
		#	[x;y] = [ [ cos(q) , sin(q) ] ; 
		#			  [-sin(q) , cos(q) ] ] * [ x ; y ]
		# Circle : x**2 + y**2 = r**2
		# Line : y = (m*z[0] + b[0] - z[1])/sqrt(1 + m**2) (no dependence on x - as expected)

		# With this simplified scenario, we can easily identify the points (x,y) where the line y = B
		# intersects the circle x**2 + y**2 = r**2
		# In our transformed coordinates, only keeping the positive root, this is:
		b, m = a[0:2]


		B = ( m * z[0] + b - z[1] ) / np.sqrt( 1 + m**2 )
		x_t = np.sqrt( r ** 2 - B ** 2 )
		y_t = B

		# TODO:// replace the fixed linear baseline with linear approximations near the intersection points

		# For contact angle, want interior angle, so look at vector in negative x direction 
		# (this is our baseline)
		v1 = [-1 , 0]

		# Now get line tangent to circle at x_t, y_t
		if B != 0:
			slope = - x_t / y_t
			v2 = np.array( [ 1 , slope ] )
			v2 = v2 / np.linalg.norm ( v2 )
			if B < 0: # We want the interior angle, so when the line is above the origin (into more negative y), look left 
				v2 = -v2
		else:
			v2 = [0 , 1]

		ϕ = { i : np.arccos ( np.dot ( v1 , v2 ) ) * 360 / 2 / np.pi for i in ['l','r'] }
		bWidth = 2 * x_t

		V = 2/3 * np.pi * r ** 3  + np.pi * r ** 2 * B - np.pi * B ** 3 / 3

		# Fitted circle
		theta = np.linspace(0,2 * np.pi,num = 100)
		x = z[0] + r * np.cos(theta)
		y = z[1] + r * np.sin(theta)
		imAxes.plot(x,y,'r-')

	else:
		raise Exception('Unknown fit type! Try another.')

	# FI FITTYPE	

	print(f'At time { j * everyNSeconds }: \t\t Contact angle left (deg): {ϕ["l"] : 6.3f} \t\t Contact angle right (deg): {ϕ["r"] : 6.3f} \t\t Contact angle average (deg): {(ϕ["l"]+ϕ["r"])/2 : 6.3f} \t\t Baseline width (px): {bWidth : 4.1f}')
	angles += [ (ϕ['l'],ϕ['r']) ]
	time += [ j * everyNSeconds ]
	baselineWidth += [ bWidth ]
	volumes += [ V ]

	# Format the plot nicely
	imAxes.set_xlim(cropPoints[0:2])
	imAxes.set_ylim(cropPoints[-1:-3:-1])
	plt.draw()
	plt.pause(0.1)

# END LOOP THROUGH IMAGES

if video:
	fig, ax1 = plt.subplots(figsize = (5,5))
	color = 'black'
	ax1.set_xlabel('Time [s]')
	ax1.set_ylabel('Contact Angle [deg]', fontsize = 10,color = color)
	ax1.plot(time,angles, marker = '.',markerfacecolor = color,markeredgecolor = color,markersize = 10
			 , linestyle = None)
	ax1.tick_params(axis = 'y', labelcolor = color)

	ax2 = ax1.twinx()
	color = 'red'
	ax2.set_ylabel('Baseline width [-]', fontsize = 10,color = color)
	ax2.plot(time,baselineWidth,marker = '.',markerfacecolor = color,markeredgecolor = color,markersize = 10 
			 , linestyle = 'None')
	ax2.tick_params(axis = 'y', labelcolor = color)

	plt.tight_layout()
	plt.draw()

if '\\' in image:
	parts = image.split('\\')
else:
	parts = image.split('/')
path = '/'.join(parts[:-1]) #Leave off the actual file part
filename = path + f'/results_{parts[-1]}.csv'

print(f'Saving the data to {filename}')

with open(filename,'w+') as file:
	file.write('Time,' + ",".join([str(t) for t in time]))
	file.write('\n')
	file.write('Left angle,' + ",".join([str(s[0]) for s in angles]))
	file.write('\n')
	file.write('Right angle,' + ",".join([str(s[1]) for s in angles]))
	file.write('\n')
	file.write('Average angle,' + ",".join([str((s[1]+s[0])/2) for s in angles]))
	file.write('\n')
	file.write('Baseline width,' + ",".join([str(s) for s in baselineWidth]))
	file.write('\n')
	file.write('Volume,' + ",".join([str(s) for s in volumes]))