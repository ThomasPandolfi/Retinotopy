from matplotlib import pyplot as plt
import numpy as np
import nibabel as nb
pol = nb.load('./rh_highestsig/aver_rhpolar_angle.nii.gz').get_fdata()
R2 = nb.load('./rh_highestsig/aver_rhR2.nii.gz').get_fdata()
polar = pol[R2 > 0.3]


N = 12
bottom = 0
max_height = 4

theta = np.linspace(-1 * np.pi, np.pi, N, endpoint=True)
polbins = plt.hist(polar, bins=theta)
width = (2*np.pi) / N	
ax = plt.subplot(111, polar=True)
bars = ax.bar((polbins[1] + (np.pi / N))[:-1], polbins[0],width=width)#, bottom=bottom)
# Use custom colors and opacity
for r, bar in zip((polbins[1] + (np.pi / N))[:-1], bars):
	bar.set_facecolor(plt.cm.jet(r / 10.))
	bar.set_alpha(0.8)


            
