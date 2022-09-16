%% Optimized Deep-E simulation for L2 arrays


addpath('C:\Users\Lab\Box\3D deep-E\Example code\3D Deep-E example code\k-Wave');

clc;
clear all;
% load the initial pressure distribution from an image and scale


directory = 'C:\Users\Lab\Box\3D deep-E\Example code\3D Deep-E example code\vessel_images';


n_img = 60; % the total number of IMG file will be input to the simulation


    figure;

    for ii = 1:n_img
        
        iii = 1;

        file_path = [directory,'\IMG',num2str(ii),'_p0.mat'];
        raw_data = load(file_path);
        size_r = size(raw_data.p0,3);
                
            
        
        for jj = 1:size_r




            p_r = raw_data.p0(:,:,jj);
            % resize the input source
%             p_r = p_r(51:250,51:250);

            p_r = imresize(p_r,[300 500]);

%             p_r_r = imresize(p_r,[500 500]);
            % add gaussian blur
            p_r = imgaussfilt(p_r,2);
            

            p_r_r = imadjust(p_r, [0 0.5]);

%                p_r_r = p_r;
          
% figure;
%                         imshow(p_r,[]);

            h1 = 101;  % index of the middle element of the transducer in the 1st frame position
            h2 = 600; % index of the middle element of the transducer in the last frame position
            final_sum_sensor = zeros((h2-h1)+1, 512);


            tic % calculate running time
            %%%%-----Define the Grid-----%%%%
            % create the computational grid
            PML_size = 20;              % size of the PML in grid points
            Nx = 1000;    % number of grid points in the x direction
            Ny = 740;    % number of grid points in the y direction
            dx = 0.1e-3;                % grid point spacing in the x direction [m]
            dy = 0.1e-3;                % grid point spacing in the y direction [m]
            kgrid = kWaveGrid(Nx, dx, Ny, dy);

            % define the properties of the propagation medium
            medium.sound_speed = 1540;	% [m/s]

            % create the time array
            dt = 1.1e-7; %sampling rate = 1/(central frequency*4)
            kgrid.t_array = ((1:512)-1)*1.1e-7;

            center_freq = 2.2727e6; % [Hz]
            bandwidth = 80; % [%] square root of 65%

            sensor.frequency_response(1) = (center_freq); % central frequency
            sensor.frequency_response(2) = (bandwidth); %transducer bandwidth

            %%%%-------Define the source-----%%%%
            % create initial pressure distribution using LoadImage

            %figure;imshow(p1,[]);
            p0 = zeros(Nx, Ny);
%             p0(351:850, 101:600) = p_r_r;
            p0(301:600, 101:600) = p_r_r;

            source.p0 = p0;
            %figure;imshow(source.p0,[]);

            % set the input arguements: force the PML to be outside the computational
            % input_args = {'PMLInside', false, 'PMLSize', PML_size, 'Smooth', false, 'PlotPML', false};
            input_args = {'PMLInside', false, 'PMLSize', PML_size, 'Smooth', false, 'PlotPML',false,...
                'DataCast','gpuArray-single','PlotSim',false};

            %-----------------
            sum_sensor_data = zeros((h2 - h1) + 1, 512);
            %-----------------
            radius    = 400; % elevation acoustic focus


            diameter  = 151; %element height


            grid_size = [Nx, Ny];
            mask_indicies = cell(h2 - h1 + 1, 1);
            sensor.mask = zeros(Nx, Ny);
            for i=h1:h2
                % define the sensor
                arc_pos   = [1, i]; % acr-shape element position
                focus_pos = [400, i]; % elevation acoustic focus at each scan position
                arc = makeArc(grid_size, arc_pos, radius, diameter, focus_pos, 'Plot', false);
                %disp(floor((i-j)/diameter)+1)
                mask_indicies{i - h1 + 1} = find(arc == 1);
                sensor.mask = sensor.mask + arc;
                %         figure; imshow(sensor.mask,[]);
            end
            sensor.mask(sensor.mask > 0) = 1;
            %     figure; imshow(sensor.mask,[]);
            sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
            all_mask_indicies = find(sensor.mask == 1);
            %disp(j)
            %disp(floor((h2-j)/diameter)+1)
            for i=h1:h2
                %disp(((j-h1)+1)+((i-1)*diameter))
                %disp(1+(diameter*(i-1)))
                %disp(i*diameter)
                %disp(((j-h1)+1)+((i-1)*diameter))
                [~, idx] = intersect(all_mask_indicies, mask_indicies{i - h1 + 1},'stable');
                %disp(idx)
                sum_sensor_data((i - h1 + 1),:) = gather(squeeze(sum(sensor_data(idx, :), 1)));
            end
            %figure;imshow(sum_sensor_data(:,:),[]);
            %     save ([checkpoint_dir, '/sensor_', num2str(k),'.mat'], 'sum_sensor_data');
            %     final_sum_sensor(:, :, k) = sum_sensor_data;
            %     timen = (nF - k) * toc / 60;
            %     waitbar(k / nF, h, [sprintf('%12.1f', timen) ' mins remaining;']);

%             imshow(sum_sensor_data,[]);







            savdir = 'C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Simulated_raw_data_test\';
            save_file_name = ['sim_raw_m_IMG_',num2str(ii),'_',num2str(iii)];
            save(fullfile(savdir,save_file_name),"sum_sensor_data");

            save_file_name = ['p0_IMG_',num2str(ii),'_',num2str(iii)];
            save(fullfile(savdir,save_file_name),"p_r_r");

            iii = iii + 1;



                disp(iii);
                disp(ii);
%                 pause;
        end

    end
    
%% Add EMI noise from experimental signal

clearvars;

raw_noise = load('C:\Users\Lab\Box\3D deep-E\Example code\3D Deep-E example code\experimental noise\RawNoise1').raw_noise;

directory_input = 'C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Simulated_raw_data_test';

savdir = 'C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Noise_and_Data_add_noise\Raw_noisy_data_EMI';
    
k = 1;

for j = 1 : 6

    for i = 1 : 250

%         rand_1 = randi([1 128],1);

        rand_n = randi([1 500],1,500);





        noise_output = squeeze(raw_noise(1:512,64,rand_n));
        noise_output = noise_output./max(noise_output,[],'all');

%         figure;imshow(noise_output,[]);

%         save_file_name = ['noise_output_',num2str(j),'_',num2str(i)];
%         save(fullfile(savdir,save_file_name),"noise_output");


        

        file_path = [directory_input,'\sim_raw_m_IMG_',num2str(j),'_',num2str(i),'.mat'];
        raw_data = load(file_path).sum_sensor_data';

        raw_data = raw_data./max(raw_data,[],'all');

        add_noise_data = raw_data + noise_output*0.6;

        save_file_name = ['add_noise_raw_',num2str(j),'_',num2str(i)];
        save(fullfile(savdir,save_file_name),"add_noise_data");



        disp(k);  
        k = k + 1;



    end
end
    

%% 3D focal line recon
clearvars;
% 3D RECONSTRUCTION CODE FOR L2 TRANSDUCER
% True 3D reconstruction



load L2FlashExtTrig_dualdisplay_128_noProcess;    % load the daq parameters

figure;
iii = 1;
for kk = 1:4
    for jj = 1:6
        for ii = 1:250
            
            
            load(['C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Noise_and_Data_add_noise\Raw_noisy_data_EMI\add_noise_raw_'...
                ,num2str(jj),'_',num2str(ii)]);
            
            load(['C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Simulated_raw_data_test\p0_IMG_'...
                ,num2str(jj),'_',num2str(ii)]);

           
            
            temp = zeros(500,500);
            
            temp(1:300,1:500) = p_r_r;
            temp = imresize(temp, [256 256]);
            p_r_r = uint8(temp./max(temp(:))*255);
            
%             figure;imshow(p_r_r);
%             savdir_1 = 'C:\Users\photoacoustic\Desktop\wenhan\U_net_input\Ground_truth\1_6000_6_9_12_free_enlarged';

%         
%             save_file_name_1 = ['p0_IMG_',num2str(iii)];
%         
%             save(fullfile(savdir_1,save_file_name_1),'p_r_r');
            
%             figure;imshow(add_noise_data,[]);

            rfdata = add_noise_data;
            
             
%             imshow(squeeze(rfdata),[]);
            
%            figure;imshow(squeeze(rfdata),[]);
            
           
            interval=0.1;  % distance between scan lines in elevation in mm
            SFormat.endDepth = size(rfdata,1);
            IntFac =1;
            %     Trans.frequency = 2.27;
            Trans.numelements = 1;
            fs = Trans.frequency*4*IntFac; % 30 MHz
            

            
            % Filtering from singal 2 to 4 MHz
            [c,d]=butter(2,[2/(fs/2),4/(fs/2)],'bandpass');
            rfdata= filter(c,d,rfdata,[],1);
            
%             figure;imshow(squeeze(rfdata),[]);


            rfdata = permute(rfdata, [1 3 2]);
            
            
%             figure;imshow(squeeze(rfdata),[]);
            
            
            
            
            
            Len_R = 40; % the array focal length
            Hlines = size(rfdata,3);              % number of elevation scan lines
            sos = 1.54;
            
            lAngle =80*pi/180;       % lateral critical receiving angle
            % eAngle =rad2deg(atan2(5,12)*2)*pi/180; % elevation receiving angle
            eAngle =150*pi/180; % elevation receiving angle
            
            wavelength = sos/Trans.frequency;
            zDepthOffset =31;        % in mm
            xDepthOffset =0;
            yDepthOffset =0;
            lines = 1;
            % use element location to create a 2D matrix, unit in mm
            % x0 =(0:(Trans.numelements-1))*Trans.spacingMm-Trans.spacingMm*(Trans.numelements-1)/2;
            
            
            
            % the single element location is defined at 0.2 mm
            x0 =((Trans.numelements))*Trans.spacingMm-Trans.spacingMm*(Trans.numelements)/2;
            ResFactor = 10;   %defines the resolution: how many pixels in 1 mm along axial and lateral directions
            YResFactor = 10;  %defines the resolution: how many pixels in 1 mm along elevation direction
            pixel_size = 1/ResFactor; % mm, the lateral and axial directs will have the same pixel size
            Ypixel_size = 1/YResFactor; % mm, the pixel size in elevation
            z_size = 50; Nz = floor(z_size*ResFactor); %defines the total reconstruction area in mm
            x_size = 0.1; Nx = floor(x_size*ResFactor); % define the length along lateral direction
            y_size = 50; Ny = floor(y_size*YResFactor);
            
            x_img0 =round(xDepthOffset)+(0:(Nx-1))*pixel_size-pixel_size*(Nx-1)/2;  % define reconstruction locations along lateral direction (mm)
            z_img0 =round(zDepthOffset) + (0:(Nz-1))*pixel_size ; % along axial direction
            y_img0 =round(yDepthOffset)+(0:(Ny-1))*Ypixel_size-Ypixel_size*(Ny-1)/2;  % along elevation
            
            if Nz*Nx*Ny > 1e9
                fprintf(2,'Split the recon region to multiple sections and combine with CAT\n');
                return
            end
            
            % conversion into 2D or 3D
            temp = ones(Nz,1);
            x_img=temp*x_img0;
            temp = ones(1,Nx);
            z_img =z_img0'*temp;
            y3D = zeros(Nz,Nx,Ny);
            for iH = 1:Ny
                y3D(:,:,iH)=ones(Nz,Nx)*y_img0(iH);
            end
            
            eleWidthWl1 = Trans.elementWidth / wavelength;
            eleWidthWl2 = Trans.elementWidth*0.5 / wavelength;
            
            y = (1:Hlines)-(Hlines+1)/2;
            y0 = y*interval;  % transducer positions in elevation
            pa_img = zeros(Nz,Nx,Ny);
            % rfdata(1,:,:)=0;
            
            %coherence weighting to suaqre the signal in each element
            % coherent_square = zeros(Nz,Nx,Ny);
            
            % convert CPU array into GPU array
            x_img = gpuArray(single(x_img));x0 = gpuArray(x0);z_img = gpuArray(single(z_img));
            y3D = gpuArray(single(y3D)); y0 = gpuArray(y0);pa_img = gpuArray(single(pa_img));
            
            % coherent_square = gpuArray(coherent_square); %coherence weighting
            
            x0_3D = repmat(x0,Nx,1,Nz);
            x0_3D = permute(x0_3D,[3,1,2]);
            
            x_img = repmat(x_img,1,1,lines);
            z_img = repmat(z_img,1,1,lines);
            r_all = sqrt((x_img - x0_3D).^2 + z_img.^2); %r_all = gather(r_all);
            Angle_line_all = atan(abs(x_img-x0_3D)./z_img);clear x0_3D x_img z_img
            din_all = (r_all-(Len_R./cos(Angle_line_all))); %din_all = gather(din_all);
            
            din_all(abs(din_all)<(1/ResFactor/2))=1/ResFactor/2;
            
            dist = Len_R./cos(Angle_line_all);
            
            %     h = waitbar(0,'Please wait...'); % create a waitbar
            for iH = 1:Hlines
                %         tic % calculate running time
                rfdataI = gpuArray(rfdata(:,:,iH));
                for iLine = 1:lines
                    itemp_Rf = rfdataI(:,iLine);
                    % use subfunction to avoid generating lots of 3D matrixs at the end
                    % of each loop
                    tmp = GPUReconLoop(Angle_line_all,din_all,y3D,y0,Ny,dist,sos,fs,eleWidthWl1,eleWidthWl2,SFormat,lAngle,eAngle,iH,iLine,itemp_Rf,IntFac,Receive);
                    pa_img = pa_img + tmp;
                    
                    %         coherent_square = coherent_square + tmp.^2;
                    
                end
            end
            
            
            pa_img = double(pa_img);
            
           
            %         pa_img = pa_img./max(pa_img,[],'all');
            
            pa_img = rescale(pa_img,-1,1);
            
            
            switch kk
                
                case 1
                    
                case 2;nl = 12;pa_img = awgn(pa_img,nl);
                    
                case 3;nl = 15;pa_img = awgn(pa_img,nl);
                    
                case 4;nl = 18;pa_img = awgn(pa_img,nl);
                    
            end
            

            
            for i=1:Ny
                pa_img_t(:,:,i)=abs(hilbert(pa_img(:,:,i)));
            end
            
            
            pa_img = pa_img_t./max(pa_img_t,[],'all');
            pa_img = permute(pa_img,[1 3 2]);
            pa_img = gather(pa_img);
            pa_img = resize(pa_img,[256 256]);
            pa_img = uint8(pa_img*255);
            

            
%             min(pa_img,[],'all');
            %         savdir = 'C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Reconstructed_image';
            savdir = 'C:\Users\photoacoustic\Desktop\wenhan\3D focal line training\Noise_and_Data_add_noise\Data_Gaussian_12_15_18_noise_free_enlarged_EMI_source';
            %         save_file_name = ['recon_IMG_',num2str(jj),'_',num2str(ii)];
            
            save_file_name = ['recon_noise_IMG_',num2str(iii)];
            %
            save(fullfile(savdir,save_file_name),"pa_img");
            
            
            
            
            imshow(pa_img,[]);
            %         figure;imshow(p0(:,:,1),[]);
%             pause;

            iii = iii+1;
            disp(iii);
        end
    end
    
end


function [pa_img] = GPUReconLoop(Angle_line_all, din_all,y3D, y0,Ny,dist, sos,fs,eleWidthWl1,eleWidthWl2,SFormat,lAngle,eAngle,iH,iLine,itemp_Rf,IntFac,Receive)
Angle_line = (Angle_line_all(:,:,iLine));
Angle_line2 = repmat(Angle_line,[1,1,Ny]);
din = (din_all(:,:,iLine));
din = repmat(din,[1,1,Ny]);
r3D = bsxfun(@(a,b) (a-b).^2, y3D, y0(iH));
r3D = bsxfun(@(a,b) sqrt(a.^2+b).*sign(a),din, r3D);
r3D = bsxfun(@(a,b) (a+b), r3D, dist(:,:,iLine));
%         r3D = sqrt(din.^2+(y3D-y0(iH)).^2).*sign(din)+repmat(dist(:,:,iLine),[1,1,Ny]);

hi = y3D - y0(iH);
Angle_line3 = bsxfun(@(a,b) atan(abs((a)./b)), hi, din);

idx= round((r3D/sos)*fs)-Receive(1).startDepth*4+IntFac;

TransSen1 = abs(cos(Angle_line).*(sin(eleWidthWl1*pi*sin(Angle_line))./(eleWidthWl1*pi*sin(Angle_line))));
TransSen1 = repmat(TransSen1,[1,1,Ny]);
tmp=  ((abs(hi)<0.8) & (abs(din)<0.8));
Angle_line3 = abs(1-tmp).*Angle_line3+tmp.*0.001;
tmp = (Angle_line3==0);
Angle_line3 = abs(1-tmp).*Angle_line3 + tmp.*0.01;
TransSen2 = abs(cos(Angle_line3).*(sin(eleWidthWl2*pi*sin(Angle_line3))./(eleWidthWl2*pi*sin(Angle_line3))));

inrange = ((idx >= 1) & (Angle_line2<=lAngle) & (idx < SFormat.endDepth) & (Angle_line3<=eAngle));
idx = (inrange).*idx + (1-inrange).*1; % if in the range, index = idx. otherwise index = zsamples (value = 0)
tmp = interp1(itemp_Rf,idx);
pa_img = arrayfun(@(a,b,c) a.*b.*c, tmp, TransSen1,  TransSen2);
%         save file.mat

end





