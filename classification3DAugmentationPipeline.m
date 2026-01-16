function [dataOut,info] = classification3DAugmentationPipeline(dataIn,info,inputSize,type)

dataOut = cell([size(dataIn,1),2]);
switch type
    case 'train' 
        for idx = 1:size(dataIn,1)
            temp = dataIn{idx};
            % temporary for dimension reduction
            temp=imcrop3(temp,[7 6 6 94 78 78]);
            % Add randomized rotation and scale
            tform = randomAffine3d('Scale',[0.85,1.15],...
                'Rotation',[-15 +15],...
                'XTranslation',[-15 15],...
                'YTranslation',[-15 15],...
                'ZTranslation',[-15 15]); 
            temp = imwarp(temp,tform);
            temp = imresize3(temp,inputSize(1:3));
            % Form second column expected by trainNetwork which is expected response,
            % the categorical label in this case
            dataOut(idx,:) = {temp,info.Label(idx)};%dataOut = temp;
        end
    case 'validation'
        for idx = 1:size(dataIn,1)
            temp = dataIn{idx};
            % temporary for dimension reduction
            temp=imcrop3(temp,[7 6 6 94 78 78]);
            temp = imresize3(temp,inputSize(1:3));
            % Form second column expected by trainNetwork which is expected response,
            % the categorical label in this case
            dataOut(idx,:) = {temp,info.Label(idx)};%dataOut = temp;
        end    
    case 'test'
        for idx = 1:size(dataIn,1)
            temp = dataIn{idx};
            % temporary for dimension reduction
            temp=imcrop3(temp,[7 6 6 94 78 78]);
            temp = imresize3(temp,inputSize(1:3));
            % Form second column expected by trainNetwork which is expected response,
            % the categorical label in this case
            dataOut(idx,:) = {temp,info.Label(idx)};%dataOut = temp;
        end        
    otherwise
        warning('Unexpected data type. No action performed.')
end

info=info;
end