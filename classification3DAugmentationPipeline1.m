function [dataOut] = classification3DAugmentationPipeline1(dataIn,inputSize,type)

dataOut = cell([size(dataIn,1),2]);
switch type
    case 'test'
        for idx = 1:size(dataIn,1)
            temp = dataIn{idx};
            % temporary for dimension reduction
            temp=imcrop3(temp,[7 6 6 94 78 78]);
            temp = imresize3(temp,inputSize(1:3));
            % Form second column expected by trainNetwork which is expected response,
            % the categorical label in this case
%             dataOut(idx,:) = {temp,info.Label(idx)};%dataOut = temp;
            dataOut(idx,:) = {temp};%dataOut = temp;

        end        
    otherwise
        warning('Unexpected data type. No action performed.')
end

end