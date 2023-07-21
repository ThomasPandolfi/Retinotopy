
wdir_RH = ['/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/RHCWEXP/sm0', '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/RHCCWCONT/sm0']
wdir_LH = ['/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/LHCWEXP/sm0', '/home/thomaszeph/Desktop/surfsmooth_experiment/raw/aver_2runs/LHCCWCONT/sm0']




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







	    
def polar_plot(Polar2, R2, N, subplotLOC, sv):

	#Just some parameters, N being the number of Bins to bin polar data
	N = 25
	bottom = 0
	max_height = 4
	
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
		
	plt.ylim([0, 650])
	return()		    



def SDvEccen_plot(SD, ECCEN, Xpos, Ypos, R2, subplotLOC, sv):

	## Set theme, pick colors to use
	sns.set_theme(style="whitegrid")
	plt.subplot(subplotLOC)
	SDEccen_colors = ('r','g','b', 'y')
	SDEccen_colors = ((31, 255, 215), (137, 252, 30), (255, 195, 43), (255, 64, 31), (167, 0, 255))
	SDEccen_colors = tuple([np.array(list(x))/255.0 for x in SDEccen_colors])	
	
	#For each significance value, plot the eccen vs sd plot for vertices of that significance to the sig .1 higher)
	for i, significance_value in enumerate(np.arange(0.2,0.6,0.1)):
			
		goodSD = SD.flatten()[(R2.flatten() > sv) & (R2.flatten() < sv + 0.1)]
		goodEccen = np.sqrt((Xpos.flatten()**2) + (Ypos.flatten() ** 2))[(R2.flatten() > sv) & (R2.flatten() < sv + 0.1)]
				
		#goodEccen = Eccen.flatten()[R2.flatten() > significance_value]
		meanSD = [np.mean(goodSD[np.round(goodEccen) == x]) for x in range(0, 15)]
		#plt.scatter(Eccen.flatten()[R2.flatten() > 0.3], SD.flatten()[R2.flatten() > 0.3]) 
		plt.plot(list(range(0, 15)), meanSD, c=SDEccen_colors[i], marker='o')
	
	#Plot details
	plt.xlabel('Eccentricity (deg)')
	plt.ylabel('Std of Gauss (deg)')
	plt.title('Eccentricity vs size')
	plt.legend(set(np.round(np.arange(0.2,0.6,0.1),1)))
	#SD.flatten()[R2.flatten() > 0.3][Eccen == 1]
	print(list(goodEccen))


