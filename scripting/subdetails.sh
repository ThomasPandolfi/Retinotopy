actualOrderPY='cw,ccw,exp,cont'
actualNumberingPY='[9, 11, 13,15]'

actualOrder=('cw' 'ccw' 'exp' 'cont' 'combined' 'cwccw' 'expcont' 'cwexp')
actualNumbering=(009 011 013 015 comb 00wedges 00rings 00cwexp)

pyR2='(009 011 013 015 comb 00wedges 00rings 00cwexp)'

fresurName='SMTEST'
niiLocation='/home/thomaszeph/Desktop/Stacey_MRI/test/analysis/main'
#niiLocation='/home/thomaszeph/Desktop/SUBJECT04/MRI/Pre/analysis/main'
#niiLocation='/home/thomaszeph/Desktop/SUBJECT01/MRI/Pre/analysis/main'
combinedFolder='comb'
upsampled=0
isCombined=1
isIndividual=1
tcAlreadyCreated=0
otherCombinations=1
numCycles=4


GauSigTemp=70
GauSigSpat=3
LFcut=0.001
HFcut=0.30


if ((numCycles == 4)); then

	OneSessionTR=382
	AllSessionTR=1528
	HalfSessionTR=764
	
elif ((numCycles == 6)); then
	OneSessionTR=568
	AllSessionTR=2272
	HalfSessionTR=1136
	
fi


HighResDimension=300
PyprfAnalysisNamePrefix='DF_r'
sysColorMap='jet'
export sysColorMap
export pyR2
export isIndividual
export HalfSessionTR
export otherCombinations
export HighResDimension
export tcAlreadyCreated
export PyprfAnalysisNamePrefix
export LFcut
export HFcut
export GauSigTemp
export GauSigSpat
export isCombined
export actualOrder
export actualNumbering
export fresurName
export niiLocation
export upsampled
export actualOrderPY
export actualNumberingPY
export AllSessionTR
export OneSessionTR





