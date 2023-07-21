% List of open inputs



%base_stem = '/home/thomaszeph/Desktop/SUBJECT01/MRI/Post/analysis/main/'
base_stem = strcat(getenv('niiLocation'), '/')

folda=['009'; '011'; '013'; '015'];


spm('defaults', 'FMRI');


%Standard combination

if str2num(getenv('isCombined'))==1


    if isfile(strcat(getenv('niiLocation'), '/comb/rcombf.nii'))
    	disp('already found combf, not reslicing again')
    
    else
	    
	disp('Combined Nii reslicing')
	numTR = str2num(getenv('AllSessionTR'));
	niichar = cell(numTR,1);
	CURNI = strcat(base_stem, 'comb/', 'combf.nii,');
	for x=1:numTR;
	    niichar(x, 1) = {strcat(CURNI, num2str(x))};
	end
	
	
	
	disp('Realign and reslice the combination')
    	load gg.mat;
    	%matlabbatch{1,1}.spm.spatial.realign.estwrite.data{1,1} = niichar;
    	spm_realign(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    	spm_reslice(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    
	
	
    end
    
    
    
    
    
    
end 




%% Other Combinations
disp('Checking Other Combinations')
if str2num(getenv('otherCombinations'))==1


    if isfile(strcat(getenv('niiLocation'), '/00wedges/r00wedgesf.nii'))
        disp('already found wedges, not reslicing again')
    else
    disp('Combined Nii reslicing')
    numTR = str2num(getenv('HalfSessionTR'));
    niichar = cell(numTR,1);
    CURNI = strcat(base_stem, '00wedges/', '00wedgesf.nii,');
    for x=1:numTR;
        niichar(x, 1) = {strcat(CURNI, num2str(x))};
    end

    disp('realign and reslice CWCCW')
    load gg.mat;
    %matlabbatch{1,1}.spm.spatial.realign.estwrite.data{1,1} = niichar;
    spm_realign(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    spm_reslice(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);

    end

    if isfile(strcat(getenv('niiLocation'), '/00rings/r00ringsf.nii'))
        disp('already found rings, not reslicing again')
    else
    %doing the rings
    numTR = str2num(getenv('HalfSessionTR'));
    niichar = cell(numTR,1);
    CURNI = strcat(base_stem, '00rings/', '00ringsf.nii,');
    for x=1:numTR;
        niichar(x, 1) = {strcat(CURNI, num2str(x))};
    end

    disp('realign and reslice contexp')
    load gg.mat;
    %matlabbatch{1,1}.spm.spatial.realign.estwrite.data{1,1} = niichar;
    spm_realign(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    spm_reslice(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    end




    if isfile(strcat(getenv('niiLocation'), '/00cwexp/r00cwexpf.nii'))
        disp('already found cwexp, not reslicing again')
    else
    %doing cwexp
    numTR = str2num(getenv('HalfSessionTR'));
    niichar = cell(numTR,1);
    CURNI = strcat(base_stem, '00cwexp/', '00cwexpf.nii,');
    for x=1:numTR;
        niichar(x, 1) = {strcat(CURNI, num2str(x))};
    end

    disp('realign and reslice cwexp')
    load gg.mat;
    %matlabbatch{1,1}.spm.spatial.realign.estwrite.data{1,1} = niichar;
    spm_realign(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    spm_reslice(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    end

end



%% Individual Runs
if str2num(getenv('isIndividual'))==1
    disp('Individual Nii one at a time')
    numTR = str2num(getenv('OneSessionTR'));
    niichar = cell(numTR,1);
    for fol = 1:4
        niichar = cell(numTR,1);
        disp(strcat('Reslicing set: ', num2str(fol)))
        CURNI = strcat(base_stem, folda(fol,:), '/', folda(fol,:), 'f.nii,');
        
        for x=1:numTR;
            niichar(x, 1) = {strcat(CURNI, num2str(x))};
        end
    
        load gg.mat;
        %matlabbatch{1,1}.spm.spatial.realign.estwrite.data{1,1} = niichar;
        spm_realign(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
        spm_reslice(niichar, matlabbatch{1,1}.spm.spatial.realign.estwrite.eoptions);
    end

end





%nrun=4;
%jobfile = {'/home/thomaszeph/Desktop/scripting/realign_script_job.m'};
%jobs = repmat(jobfile, 1, nrun);
%inputs = cell(0, nrun);





