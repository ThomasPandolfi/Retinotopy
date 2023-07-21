


#Set up Freesurfer and give permissions
export FREESURFER_HOME=/usr/local/freesurfer/7.3.2
export SUBJECTS_DIR=/usr/local/freesurfer/7.3.2/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh
sudo chmod -R a+w $SUBJECTS_DIR


## Not this one for pyprf but still needed for fresurfer stuff
#dcmunpack -src ./predcm -targ analysis -run 7 main nii T1.nii -run 9 main nii f.nii -run 11 main nii f.nii -run 13 main nii f.nii -run 15 main nii f.nii -run 17 main nii f.nii
#################################




#dcmunpack -src ./predcm -targ analysis -run 9 main nii 009f.nii -run 11 main nii 011f.nii -run 13 main nii 013f.nii -run 15 main nii 015f.nii

#For T1
#dcmunpack -src ./predcm -targ T1 -run 7 main nii T1.nii
#recon-all -s SMTEST -i ./analysis/main/007/T1.nii -all

#Get environment variables from subdetails file
source subdetails.sh
cd $niiLocation
cd ../..

#put subjectname in analysis
#put sessid in Post
echo analysis > sessid
cd analysis
echo ${fresurName} > subjectname
cd ..
mkdir ${niiLocation}/comb
mkdir ${niiLocation}/00rings
mkdir ${niiLocation}/00wedges
mkdir ${niiLocation}/00cwexp

#cd ..
#preproc-sess -sf ./sessid -fsd main -surface ${fresurName} lhrh -mni305 -fwhm 2 -per-run

#preproc-sess -sf ./sessid -fsd main -surface ${fresurName} lhrh -fwhm 0 -per-run

cd /home/thomaszeph/Desktop/scripting/

python3 combination.py


matlab -nodisplay -nosplash -nodesktop -r "run('./realign_script.m');exit;"

python3 norm.py


#Copy directory structure to the main analysis folder
cp /home/thomaszeph/Desktop/scripting/pyprf ${niiLocation}/ -r


#Create appropriate parameters for the analysis
for value in 0 1 2 3 4 5 6 7; do
#cp ~/Desktop/scripting/config.csv ${niiLocation}/pyprf/${actualNumbering[${value}]}
ConfigLocation=${niiLocation}/pyprf/${actualNumbering[${value}]}/config.csv

#Move into the pyprf and run specific section
cd ${niiLocation}/pyprf/${actualNumbering[${value}]}
mkdir tc
echo varNumX = 23 >> $ConfigLocation
echo varNumY = 23 >> $ConfigLocation
echo varNumPrfSizes = 14 >> $ConfigLocation
echo varExtXmin = -11 >> $ConfigLocation
echo varExtXmax = 11 >> $ConfigLocation
echo varExtYmin = -11 >> $ConfigLocation
echo varExtYmax = 11 >> $ConfigLocation
echo varPrfStdMin = 0.2 >> $ConfigLocation
echo varPrfStdMax = 10 >> $ConfigLocation
echo varTr = 0.650 >> $ConfigLocation
echo varVoxRes = 2.4 >> $ConfigLocation
echo varSdSmthTmp = 0 >> $ConfigLocation
echo varSdSmthSpt = ${GauSigSpat} >> $ConfigLocation
echo lgcLinTrnd = False >> $ConfigLocation
echo varPar = 4 >> $ConfigLocation
echo varVslSpcSzeX = ${HighResDimension} >> $ConfigLocation
echo varVslSpcSzeY = ${HighResDimension} >> $ConfigLocation
#lstPathNiiFunc = ['']    
echo lstPathNiiFunc = [\'${niiLocation}/${actualNumbering[${value}]}/${PyprfAnalysisNamePrefix}${actualNumbering[${value}]}f.nii\'] >> $ConfigLocation
#strPathNiiMask = ''
echo strPathNiiMask = \'${niiLocation}/masks/brain.nii.gz\' >> $ConfigLocation
echo strPathOut = \'results\' >> $ConfigLocation
echo strVersion = \'cython\' >> $ConfigLocation

#lgcCrteMdl = False      create time course models
if (($tcAlreadyCreated == 1)); then echo lgcCrteMdl = False >> $ConfigLocation; else echo lgcCrteMdl = True >> $ConfigLocation; fi

#lstPathPng = ['']
#echo lstPathPng = [\'${niiLocation}/pyprf/${actualOrder[${value}]}/frame_\'] >> $ConfigLocation
echo lstPathPng = [\'/home/thomaszeph/Desktop/scripting/pyprf/${actualOrder[${value}]}/frame_\'] >> $ConfigLocation


echo varStrtIdx = 1 >> $ConfigLocation
#varZfill = 4 or 3
if (($value > 3 )); then
echo varZfill = 4 >> $ConfigLocation
else
echo varZfill = 3 >> $ConfigLocation
fi

#strPathMdl = './tc/prf_tc'
echo strPathMdl = \'./tc/prf_tc\' >> $ConfigLocation

#lgcHdf5 = False
if (($value == 4)); then
echo lgcHdf5 = True >> $ConfigLocation
else
echo lgcHdf5 = False >> $ConfigLocation
fi



#Run the PYPRF analysis
pyprf -config $ConfigLocation



#Move everything to a foldername with the appropriate parameters
tempfile=${GauSigSpat}_${GauSigTemp}_${LFcut}_${HFcut}_${HighResDimension}
mkdir ${niiLocation}/pyprf/${actualNumbering[${value}]}/$tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/config.csv $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_eccentricity.nii.gz $tempfile 
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_PE_01.nii.gz $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_polar_angle.nii.gz $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_R2.nii.gz $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_SD.nii.gz $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_x_pos.nii.gz $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/results_y_pos.nii.gz $tempfile
mv ${niiLocation}/pyprf/${actualNumbering[${value}]}/tc $tempfile




done

#source /home/thomaszeph/Desktop/scripting/fanchange.sh


echo $tempfile >> ${niiLocation}/pyprf/${actualNumbering[${value}]}/log






#preproc-sess -d ${niiLocation} -s ${fresurName} -fsd main -surface ${fresurName} lhrh -mni305 -fwhm ${GauSigSpat} -per-run

