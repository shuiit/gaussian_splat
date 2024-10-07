function tbl = get_3d_pts(filename,imageid)


fid = fopen(filename, 'r');
header = {'POINT3D_ID', 'X', 'Y', 'Z','imageid','pt_idx'};
% Initialize a line counter
tbl = array2table(nan(0,length(header)),'VariableNames',header)
j = 1;
% Read each line of the file
while ~feof(fid) 
    % Read a single line
    line = fgetl(fid);
    
    % Check if the line is not a comment and is an odd line (1st, 3rd, 5th, ...)
    if ~startsWith(line, '#')
        try
        % Read the line into variables based on the format
        temp = textscan(line, '%f', 'Delimiter', ' ');
        if sum(temp{1}(9:2:end ) == imageid)>0
            img_idx= find(temp{1} == imageid);
            data(j,1:length(header)) = [temp{1}(1:4)',temp{1}(img_idx(end)),temp{1}(img_idx(end)+1)];
            j = j + 1;
        end
        catch
            continue 
        end

        
    end
end
tbl = array2table(data,'VariableNames',header);
fclose(fid);
end