wdir_RH = '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/rh_highestsig'
wdir_LH = '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/lh_highestsig'

base_fname = 'aver_lh'
import numpy as np
import nibabel as nb
from matplotlib import pyplot as plt	
		
R2LH = nb.load(wdir_LH + '/' + base_fname + 'R2.nii.gz').get_fdata()
XPOSLH = nb.load(wdir_LH +  '/' + base_fname + 'x_pos.nii.gz').get_fdata()
YPOSLH = nb.load(wdir_LH + '/' + base_fname + 'y_pos.nii.gz').get_fdata()
POLARLH = nb.load(wdir_LH + '/' + base_fname + 'polar_angle.nii.gz').get_fdata()
#POLARLH = nb.load(wdir_LH + '/' + base_fname + 'polar_angle_VOLlh.nii.gz').get_fdata()
SDLH = nb.load(wdir_LH + '/' + base_fname + 'SD.nii.gz').get_fdata()
ECCENLH = nb.load(wdir_LH + '/' + base_fname + 'eccentricity.nii.gz').get_fdata()
AFFINELH = nb.load(wdir_LH+ '/' + base_fname + 'R2.nii.gz').affine
	
	
base_fname = 'aver_rh'
R2RH = nb.load(wdir_RH + '/' + base_fname + 'R2.nii.gz').get_fdata()
XPOSRH = nb.load(wdir_RH +  '/' + base_fname + 'x_pos.nii.gz').get_fdata()
YPOSRH = nb.load(wdir_RH + '/' + base_fname + 'y_pos.nii.gz').get_fdata()
POLARRH = nb.load(wdir_RH + '/' + base_fname + 'polar_angle.nii.gz').get_fdata()
#POLARRH = nb.load(wdir_RH + '/' + base_fname + 'polar_angle_VOLrh.nii.gz').get_fdata()
SDRH = nb.load(wdir_RH + '/' + base_fname + 'SD.nii.gz').get_fdata()
ECCENRH = nb.load(wdir_RH + '/' + base_fname + 'eccentricity.nii.gz').get_fdata()
AFFINERH = nb.load(wdir_RH + '/' + base_fname + 'R2.nii.gz').affine

def polar_plot(Polar2, R2, N, subplotLOC, sv):
	N = 25
	bottom = 0
	max_height = 4
	
	theta = np.linspace(-1 * np.pi, np.pi, N, endpoint=True)
	polbins = np.histogram(Polar2.flatten()[R2.flatten() > sv], bins=theta)
	print(np.sum(polbins[0]))
	width = (2*np.pi) / N	
	ax = plt.subplot(subplotLOC, polar=True)
	#polbins = np.histogram(Polar.flatten()[R2.flatten() > sv], bins=theta)
	#polbins = plt.hist(Polar.flatten()[R2.flatten() > sv], bins=theta)
	bars = ax.bar((polbins[1] + (np.pi / N))[:-1], polbins[0], width=width)#, bottom=bottom)
	# Use custom colors and opacity
	#print(polbins)
	
	print(polbins)
	
	for thet, bar in zip((polbins[1] + (np.pi / N))[:-1], bars):
		#bar.set_facecolor(cmap(((thet + np.pi) / (2*np.pi))))
		bar.set_facecolor(plt.cm.rainbow(((thet + np.pi) / (2*np.pi))))
		bar.set_alpha(0.8)
	plt.ylim([0, 900])
	return()		
		
plt.figure(1)
		
R2 = np.vstack((R2LH, R2RH))
Xpos = np.vstack((XPOSLH, XPOSRH))
Ypos = np.vstack((YPOSLH, YPOSRH))
Polar = np.vstack((POLARLH,POLARRH))
SD = np.vstack((SDLH, SDRH))
Eccen = np.vstack((ECCENLH,ECCENRH))

Xpos = np.cos(Polar.flatten()) * Eccen.flatten()
Ypos = np.sin(Polar.flatten()) * Eccen.flatten()
R2 = R2.flatten()

Polar = np.arctan2(Ypos.flatten(), Xpos.flatten())
Eccen = np.sqrt((Xpos.flatten() ** 2) + (Ypos.flatten() ** 2))
		



polar_plot(np.round(Polar,4), R2, 12, 111, 0.7)  

plt.show()
