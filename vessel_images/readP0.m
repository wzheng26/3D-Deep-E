clc;
clear all;
% load the initial pressure distribution from an image and scale

currentdirectory = pwd;


all_files = dir; % list folder contents
all_dir = all_files([all_files(:).isdir]); % select the folder
num_dir = numel(all_dir(3:end));


for i = 1:num_dir
    myFolder = [currentdirectory,'\image',num2str(i),'\original_image']; % Define your working folder
    % myFolder = 'E:\ZWH\Deep_learning\Imaging_depth_improvement\Image_data\vessel images\image1\original_image'; % Define your working folder

    filePattern = fullfile(myFolder, '*.jpg');
    imageFiles = dir(filePattern);
    n=1;


    

    for k = 21:290
        baseFileName = imageFiles(k).name;
        fullFileName = fullfile(myFolder, baseFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);
        % create initial pressure distribution using LoadImage
        p1 =  loadImage(fullFileName);

        if size(p1,1) == 301 || size(p1,1) == 376 || size(p1,1) == 361

            p1(isnan(p1))=1;
            p1 = 1-p1;
            p0(:,:,n)=p1;
            n=n+1;

            if n>250

                break

            end

            

        else 



        end

    end

    p0_name = ['IMG',num2str(i),'_p0'];

    save(p0_name,'p0');

    clear p0




end


figure;
for i = 1:247
    imshow(noisy_raw_m(:,:,i),[]);
end

