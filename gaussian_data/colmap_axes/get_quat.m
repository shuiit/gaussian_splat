function [tbl,pts_2d] = get_quat(filename,image_id)
pts_2d = [];
formatSpec = '%d %f %f %f %f %f %f %f %d %s';

fid = fopen(filename, 'r');
% Initialize a line counter
lineCount = 0;
tbl = array2table(nan(0,10),'VariableNames',{'IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME'})
% Read each line of the file
while ~feof(fid)
    % Read a single line
    line = fgetl(fid);
    
    % Check if the line is not a comment and is an odd line (1st, 3rd, 5th, ...)
    if ~startsWith(line, '#')
        lineCount = lineCount + 1; % Increment line counter
        temp = textscan(line, formatSpec, 'Delimiter', ' ');

        % Only process odd lines
        if mod(lineCount, 2) == 1
            
            % Read the line into variables based on the format
            tbl_row = cell2table(temp(1:10),'VariableNames',{'IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME'});
            % Append the parsed data to the cell array
            tbl = [tbl;tbl_row];

            
            
            if temp{1} == image_id
                line = fgetl(fid);
                 lineCount = lineCount + 1; % Increment line counter
                pts_2d = textscan(line, '%f', 'Delimiter', ' ');
            end
       
        end

        
    end
end
fclose(fid);
end