#sample R2 data convolved with Polar/Eccen/X and Y position
#Displays significant voxels in a plot
import os
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nb
import sys
from scipy.special import rel_entr
import matplotlib
import seaborn as sns
markerSize = 16

#Working directory with nii result files
wdir = '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/cwexp_demeanedfirst/001/sm2_pyprf/results_rh/'
#wdir_RH = '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aight/sm3/rh'
#wdir_LH = '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aight/sm3/lh'
which_plots = [[0,1,2,3], [4,5,6,7]]
which_plots = [[0]]
#runFolders = os.environ['pyR2'][1:-1].split(' ')
runFolders = 'cwexp'
sysColorMap = 'jet'
sns.set_theme(style="dark")





#wdir_RH = ['/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/RHCWEXP/sm0', '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/RHCCWCONT/sm0']
#wdir_LH = ['/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/LHCWEXP/sm0', '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/LHCCWCONT/sm0']




## Hemi_dir should be a two value array with the CWEXP and CCWCONT files  or any two combo files from a single hemisphere
#hemi is the hemiphere
def aver_results(hemi_dir, hemi):

	base_fname = 'results_'

	NM = ['R2', 'x_pos', 'y_pos', 'polar_angle', 'SD', 'eccentricity']
	
	for x in NM:
	 	
		A = nb.load(hemi_dir[0] + '/' + base_fname + x + '.nii.gz').get_fdata()
		B = nb.load(hemi_dir[1] + '/' + base_fname + x + '.nii.gz').get_fdata()
		C = nb.load(hemi_dir[0] + '/' + base_fname + 'R2.nii.gz').affine
		AR2 = nb.load(hemi_dir[0] + '/' + base_fname + 'R2.nii.gz').get_fdata()
		BR2 = nb.load(hemi_dir[1] + '/' + base_fname + 'R2.nii.gz').get_fdata()
		#av = np.mean((A,B), axis=0)
		av = np.sum(((A * [AR2 >= BR2]) + (B * [BR2 > AR2])), axis=0)
		im = nb.Nifti1Image(av, C)
		nb.save(im, 'aver_' + hemi + x + '.nii.gz')

	return()


#os.environ['GauSigSpat']
#run_extension =  os.environ['GauSigSpat'] + '_' + os.environ['GauSigTemp'] + '_' + os.environ['LFcut'] + '_' + os.environ['HFcut'] + '_' + os.environ['HighResDimension']

import matplotlib.colors
#### Create useful colormap

#cvals = [0, 0.45, 0.69, 1, 1.28]
#ccolors = [(255,0,0), (255, 115, 0), (255, 255, 0), (153, 255, 0), (0, 255, 0)]
#ccolors = [(1,0,0), (1,0.5,0), (1,1,0), (0.5,1,0), (0,1,0)]



## Regular Polar OLD
#ccolors = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.45098039215686275, 0.00392156862745098), (1.0, 1.0, 0.0), (0.13725490196078433, 1.0, 0.00784313725490196), (0.00784313725490196, 0.5372549019607843, 1.0), (0.011764705882352941, 0.058823529411764705, 1.0), (0.0, 0.0, 0.4980392156862745)]
#cvals = [-3.062722057667684, -1.4619646807839874, -1.176525064562243, -0.48250397814598145, 0.5, 1.1213265343248744, 1.4304383036541393, 2.8198485246232607]


### SEMI COLOR FOR POLARS
#ccolors = [(1.0, 0.0, 0.0), (1.0, 0, 0.5), (1.0, 0.0, 1.0), (0.4980392156862745, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.5019607843137255, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.5019607843137255), (0.0, 1.0, 0.0), (0.5019607843137255, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5019607843137255, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.4980392156862745), (1.0, 0.0, 1.0), (0.4980392156862745, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.5019607843137255, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.5019607843137255), (0.0, 1.0, 0.0), (0.5019607843137255, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5019607843137255, 0.0), (1.0, 0.0, 0.0)]
#cvals = [-3.1415, -3.0509328842163086, -2.618, -2.356, -2.094, -1.833, -1.571, -1.309, -1.047, -0.785, -0.523, -0.26, 0, 0.262, 0.523, 0.785, 1.047, 1.309, 1.571, 1.833, 2.094, 2.356, 2.618, 2.88, 3.1415927410125732]



##Regular Color for polars
ccolors = [(1.0, 0.0, 0.0), (1.0, 0.5019607843137255, 0.0), (1.0, 1.0, 0.0), (0.5019607843137255, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.5019607843137255), (0.0, 1.0, 1.0), (0.0, 0.5019607843137255, 1.0), (0.0, 0.0, 1.0), (0.5019607843137255, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.5019607843137255), (1.0, 0.0, 0.0)]


cvals = [-3.14, -2.6175, -2.095, -1.571, -1.047, -0.523, 0, 0.523, 1.047, 1.571, 2.095, 2.6175, 3.14]




surfnorm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(surfnorm, cvals), ccolors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


#code to sample from freesurfer colormaps into your own
#Assign dictionary A
#ccolors = []
#cvals = []
#for x in range(0, len(A)):

#	V = (np.array([[A[x]['r'], A[x]['g'], A[x]['b']]]) / 255)[0]
#	ccolors.append(tuple(V))
#	cvals.append(A[x]['val']


def polar_plot(Polar2, R2, N, subplotLOC, sv):

	#Just some parameters, N being the number of Bins to bin polar data
	N = 25
	bottom = 0
	max_height = 4
	
	
	origin = 0
	
	
	#Full range from -pi to pi
	theta = np.linspace(-1 * np.pi, np.pi, N, endpoint=True)
	#polbins = np.histogram(Polar.flatten()[R2.flatten() > sv], bins=theta)
	width = (2*np.pi) / N	
	
	#choose subplot, bin data, turn into bars
	ax = plt.subplot(subplotLOC, polar=True)
	polbins = np.histogram(Polar2.flatten()[R2.flatten() > sv], bins=theta)
	#polbins = plt.hist(Polar.flatten()[R2.flatten() > sv], bins=theta)
	bars = ax.bar((polbins[1] + (np.pi / N))[:-1], polbins[0], width=width)#, bottom=bottom)
	
	#for each bar pick the color
	for thet, bar in zip((polbins[1] + (np.pi / N))[:-1], bars):
		bar.set_facecolor(cmap(((thet + np.pi) / (2*np.pi))))
		bar.set_facecolor(plt.cm.rainbow(((thet + np.pi) / (2*np.pi))))
		bar.set_alpha(1)
		
	plt.ylim([0, np.max(polbins[0])])
	return()		    

def eccen_plot(Xpos, Ypos, R2, N, subplotLOC, sv):
	goodEccen = np.sqrt((Xpos.flatten()**2) + (Ypos.flatten() ** 2))[R2.flatten() > sv]
		
	#choose subplot, bin data, turn into bars
	ax = plt.subplot(subplotLOC)#, polar=True)
	polbins = np.histogram(Polar2.flatten()[R2.flatten() > sv], bins=theta)
	#polbins = plt.hist(Polar.flatten()[R2.flatten() > sv], bins=theta)
	bars = ax.bar((polbins[1] + (np.pi / N))[:-1], polbins[0], width=width)#, bottom=bottom)
	
	return(1)
	

def SDvEccen_plot(SD, Eccen, Xpos, Ypos, R2, subplotLOC):

	## Set theme, pick colors to use
	R2_bounds = np.arange(0.1,0.6,0.1)
	
	sns.set_theme(style="whitegrid")
	plt.subplot(subplotLOC)
	SDEccen_colors = ('r','g','b', 'y')
	SDEccen_colors = ((31, 255, 215), (137, 252, 30), (255, 195, 43), (255, 64, 31), (167, 0, 255))
	SDEccen_colors = ((0, 194, 227), (39,83,97), (0, 173, 119))
	#SDEccen_colors = ((255, 103, 0), (97, 62, 39), (173, 19,0))
	SDEccen_colors = ((222, 0, 0), (145, 0, 2), (94, 0, 0))
	SDEccen_colors = ((252,0,0), (162,0,0), (78,0,0), (0,0,0))
	SDEccen_colors = ((0,214,214), (0,61,158), (6,0,189), (59,0,87), (33,1,21))
	SDEccen_colors = ((0,118,201), (0,90,92), (94,90,93), (92,79,9), (212,153,15))
	SDEccen_colors = ((1,243,255), (1,31,255), (0,0,0), (143,0,124), (255,10,0))
	
	SDEccen_colors = tuple([np.array(list(x))/255.0 for x in SDEccen_colors])	
	
	#For each significance value, plot the eccen vs sd plot for vertices of that significance to the sig .1 higher)
	for i, sv in enumerate(R2_bounds):
			
		#goodSD = SD.flatten()[(R2.flatten() > sv) & (R2.flatten() < sv + 0.1)]
		#goodEccen = np.sqrt((Xpos.flatten()**2) + (Ypos.flatten() ** 2))[(R2.flatten() > sv) & (R2.flatten() < sv + 0.1)]
		
		goodEccen = np.sqrt((Xpos.flatten()**2) + (Ypos.flatten() ** 2))[R2.flatten() > sv]		
		goodSD = SD.flatten()[R2.flatten() > sv]
		
		meanSD = [np.mean(goodSD[np.round(goodEccen) == x]) for x in range(0, 15)]
		#plt.scatter(Eccen.flatten()[R2.flatten() > 0.3], SD.flatten()[R2.flatten() > 0.3]) 
		plt.plot(list(range(0, 15)), meanSD, c=SDEccen_colors[i], marker='o')
	
	#Plot details
	plt.xlabel('Eccentricity (deg)')
	plt.ylabel('Std of Gauss (deg)')
	plt.title('Eccentricity vs size')
	plt.legend(np.round(np.arange(0.1,0.6,0.1),1))
	#SD.flatten()[R2.flatten() > 0.3][Eccen == 1]
	#print(list(goodEccen))

def size_distribution(R2, SD, Eccen, fignum, sv):
	
	import scipy.stats as stats
	plt.figure(fignum)
	sns.set_theme(style = 'darkgrid')
	#goodEccen = np.sqrt((Xpos.flatten()**2) + (Ypos.flatten() ** 2))[R2.flatten() > sv]		
	goodEccen = Eccen.flatten()[R2.flatten() > sv]
	goodSD = SD.flatten()[R2.flatten() > sv]
	
	NPM_peaks= []
	PM_peaks = []
	regular_mean = []
	xax = [a / 2.0 for a in list(range(0,20))]
	xax = list(range(0,10))
	for x in range(0, 10):
		plt.subplot(2,5,x + 1)
		abc = plt.hist(goodSD[np.round(goodEccen) == x], bins=10, density=True, edgecolor='black', facecolor='white')
		print(abc)
		#print(x, goodSD[np.round(goodEccen*2)/2.0 == x/2])
		
# non-parametric pdf
		nparam_density = stats.kde.gaussian_kde(goodSD[np.round(goodEccen) == x])
		den = np.linspace(0, 10, 100)
		nparam_density = nparam_density(den)
		plt.plot(den, nparam_density, c='blue')
		NPM_peaks.append(np.where(nparam_density == np.max(nparam_density))[0])
		
		#plt.plot(nparam_density, den)
		
		
		# parametric fit: assume normal distribution
		loc_param, scale_param = stats.norm.fit(goodSD[np.round(goodEccen) == x])
		param_density = stats.norm.pdf(den, loc=loc_param, scale=scale_param)
		plt.plot(den, param_density, c='red')
		PM_peaks.append(np.where(param_density == np.max(param_density))[0])
		
		
		regular_mean.append(np.mean(goodSD[np.round(goodEccen) == x]))
	
	PMP = [den[b][0] for b in PM_peaks]
	NPMP = [den[b][0] for b in NPM_peaks]
	
	plt.figure(fignum + 1)
	plt.plot(xax, regular_mean, c='black', marker='o', label='Regular Mean')
	plt.plot(xax, NPMP, c='blue', marker='o', label = 'NonParametric Peak')
	plt.plot(xax, PMP, c='red', marker='o', label = 'Parametric Peak')
	plt.legend()
	plt.title('Size vs Eccentricity')
	plt.xlabel('Eccentricity')
	plt.ylabel('Size of sigma (degrees)')
	
	return()

def greysize_plot(Xpos, Ypos, SD, R2, subplotLOC, sv):
	plt.subplot(subplotLOC)
	plt.scatter(Xpos.flatten()[R2.flatten() > sv],Ypos.flatten()[R2.flatten() > sv], s=(4*SD.flatten()[R2.flatten()>sv])**2, alpha=0.3, edgecolors='black', facecolors='none')
	plt.ylim([-11,11])
	plt.xlim([-11,11])
	plt.xlabel('Degrees (horizontal)')
	plt.ylabel('Degrees (vertical)')


def colorsize_plot(Xpos, Ypos, SD, R2, subplotLOC, sv):
	plt.subplot(subplotLOC)
	#print(R2[R2.flatten() > sv])
	#print('asdasdasdasdasdasdasdasdsadasdsadasdasdasdasdsadas\n\n\n\n')
	#print(R2[R2.flatten() > sv, 0, 0]) 
	
	try:
		random_colors = list(map(plt.cm.rainbow, SD[R2.flatten() > sv,0,0]/10))
		print('Without Label')
	except:
		random_colors = list(map(plt.cm.rainbow, SD[R2.flatten() > sv]/10))
		print('With Label')
	#print(random_colors)
	plt.scatter(Xpos.flatten()[R2.flatten() > sv],Ypos.flatten()[R2.flatten() > sv], s=(4*SD.flatten()[R2.flatten()>sv])**2, alpha=0.3, edgecolors=random_colors, facecolors='none')
	
	plt.ylim([-11,11])
	plt.xlim([-11,11])
	plt.xlabel('Degrees (horizontal)')
	plt.ylabel('Degrees (vertical)')

	
#def loadData(*a, *kw):
	


def loadResults(*a, **kw):
	everything = [nb.load(kw['add_prefix'] + kw['normal_prefix'] + x + kw['end_suffix']) for x in a]
	return(everything)
	
	
#def combineResults(

def regular_dotplot(Xpos, Ypos, R2, subplotLOC, sv):


	plt.subplot(subplotLOC)
	
	
	X = Xpos.flatten()[R2.flatten() > sv]
	Y = Ypos.flatten()[R2.flatten() > sv]
	#Find number of each location
	CombinedXY = np.transpose(np.vstack((X,Y)))
	Cmapper = np.array([])
	tup = dict()
	for x in CombinedXY:
		if tuple(x) in tup:
			tup[tuple(x)] += 1
		else:
			tup[tuple(x)] = 1

	Cmapper = np.array(list(tup.keys()))
	XYC = np.transpose(np.vstack((Cmapper[:,0],Cmapper[:,1], np.array(list(tup.values())))))
		
	#Plot Results	
	sCM = 'rainbow'
	dotSize = 16
	plt.scatter(XYC[:,0], XYC[:,1], c=XYC[:,2], norm=matplotlib.colors.LogNorm(), cmap=sCM, s=dotSize)
	plt.ylim([-12,12])
	plt.xlim([-12,12])
	plt.title(specificFolder)
		
		
		
def read_label(label_dir, thresh):
    
    #Takes each line in label file
    #splits string into list
    #maps the float argument for each value
    
    with open(label_dir) as f:
        f.readline()
        f.readline()
        contents = [list(map(float, line.rstrip().split())) for line in f.readlines()]
    
    contents = np.array(contents)
    viableVertex = contents[:,0][contents[:,4] >= thresh]
    return(viableVertex)
    
    
#generic version for reading a label and parsing into usable indexer
#Locations here is simply
def label_adjuster(locations,hemisize):
	l = np.array(list(map(int, locations)))
	v = np.zeros((hemisize,1,1))
	v[l - 1] = 1
	return(v)
				

def label_adder(verts1, verts2, *third):
	
	if third == ():
		return(1 * ((verts1 + verts2) >= 1))	
	
	else:
		return(1 * ((verts1 + verts2 + third[0]) >= 1))		   
    

base_fname = 'results_'
import sys
#R2 significance. Only take R2 above a certain amount

try:
	significance = float(sys.argv[1])
except:	
	significance = 0.3


isPolar = 0
isEccen = 0
isSD = 0

#plt.figure(5)
cc = 0

which_plots = [[0,1,2,3], [4,5,6,7]]
which_plots = [[0]]
for figureSeparation in range(0,len(which_plots)):
	
	plt.figure(figureSeparation)
	#fig, axes = plt.subplots(2, 4)
	for i, plotNumber in enumerate(which_plots[figureSeparation]):
       	#plt.figure(figureSeparation)
		cc += 1
		specificFolder = 'new'
		sns.set_theme(style="dark")
		

		Positional = ['R2', 'x_pos', 'y_pos', 'polar_angle', 'eccentricity', 'SD']
		
		
		## Load data nii files
		LOC = {'add_prefix': wdir, 'normal_prefix': 'results_', 'end_suffix':'.nii.gz'}
		R2, XPOS, YPOS, POLARS, ECCEN, SD = loadResults(*Positional, **LOC)
        
		do_label = 1
		
		V1_lh_label = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/lh.V1_exvivo.label'
		V1_rh_label = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/rh.V1_exvivo.label'
		V2_lh_label = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/lh.V2_exvivo.label'
		V2_rh_label = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/rh.V2_exvivo.label'
		MT_lh_label = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/lh.MT_exvivo.label'
		MT_rh_label = '/usr/local/freesurfer/7.3.2/subjects/DB00/label/rh.MT_exvivo.label'
		#V1_lh_location = read_label(V1_lh_label, 0.1)
		#V1_lh_location = np.array(list(map(int, V1_lh_location)))
		#V1_lh_vertices = np.zeros((138839,1,1))
		#V1_lh_vertices[V1_lh_location - 1] = 1
		
		
		
		#V1_rh_location = read_label(V1_rh_label, 0.1)
		#V1_rh_location = np.array(list(map(int, V1_rh_location)))
		#V1_rh_vertices = np.zeros((137824,1,1))
		#V1_rh_vertices[V1_rh_location - 1] = 1
		
		
		V1_rh_vertices = label_adjuster(read_label(V1_rh_label, 0.4), 137824)
		V1_lh_vertices = label_adjuster(read_label(V1_lh_label,0.4), 138839)
		V2_rh_vertices = label_adjuster(read_label(V2_rh_label, 0.2), 137824)
		V2_lh_vertices = label_adjuster(read_label(V2_lh_label,0.2), 138839)
		MT_rh_vertices = label_adjuster(read_label(MT_rh_label, 0.2), 137824)
		MT_lh_vertices = label_adjuster(read_label(MT_lh_label,0.2), 138839)
		V2V1_lh_vertices = label_adder(V2_lh_vertices, V1_lh_vertices)
		V2V1_rh_vertices = label_adder(V2_rh_vertices, V1_rh_vertices)
		V2V1MT_lh_vertices = label_adder(V1_lh_vertices, V2_lh_vertices, MT_lh_vertices)
		V2V1MT_rh_vertices = label_adder(V1_rh_vertices, V2_rh_vertices, MT_rh_vertices)
		
		if do_label:
		    filecombo = lambda a : np.hstack((a[0].get_fdata()[V1_lh_vertices >= 1], a[1].get_fdata()[V1_rh_vertices >= 1]))
		
		else:
		    
		    #Combine the data from hemispheres
		    filecombo = lambda a : np.vstack((a[0].get_fdata(), a[1].get_fdata()))
            
            
		## This convention is legacy, useful for the R2_combination.py file but kept here for code simplicity
		vals_2_combine = [R2, XPOS, YPOS, POLAR, SD, ECCEN]
		R2, Xpos, Ypos, Polar, SD, Eccen = [filecombo(f) for f in vals_2_combine]
		
		
		#Polar = np.arctan2(Ypos.flatten(), Xpos.flatten())
		#Eccen = np.sqrt((Xpos.flatten() ** 2) + (Ypos.flatten() ** 2))
		
		#Xpos = np.cos(Polar.flatten()) * Eccen.flatten()
		#Ypos = np.sin(Polar.flatten()) * Eccen.flatten()
		#R2 = R2.flatten()
		
	
		regular_dotplot(Xpos, Ypos, R2, 311, significance)    
		polar_plot(np.round(Polar,4), R2, 12, 312, significance)  
		SDvEccen_plot(SD, Eccen, Xpos, Ypos, R2, 313)	
				
		plt.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.07)
	
	




 
plt.figure(80)
greysize_plot(Xpos, Ypos, SD, R2, 211, significance)
colorsize_plot(Xpos,Ypos, SD, R2, 212, significance)		
plt.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.03)
	
sns.set_theme(style="whitegrid")


print(R2.shape)
print('Where > 0.3')
print(np.sum(R2>0.3))
print('Where > 0.5')
print(np.sum(R2 > 0.5))
	

plt.show()	

