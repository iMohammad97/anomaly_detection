function [disc_dist_a, disc_loc_a, disc_nn_loc_a, disc_times, disc_misses] = MERLIN3(a, minLength, maxLength, r, display_meta_info)
% Eamonn Keogh, Takaaki Nakamura, Makoto Imamura, conceived of the idea of MERLIN
% Kaveh Kamgar implemented the function discord_discovery_gemm
% Ryan Mercer implemented the function MERLIN

%Last updated 2021-07-29

if nargin == 4
    display_meta_info = false;
end
numLengths = maxLength-minLength+1;
disc_dist_a = zeros(numLengths, 1);
disc_loc_a = zeros(numLengths, 1);
disc_nn_loc_a = zeros(numLengths, 1);

disc_times = zeros(numLengths, 1);
disc_misses = zeros(numLengths, 1);
lengths = minLength:maxLength;
for lengthIndex = 1:numLengths
    tic;
    while true
        disp('---');
        [temp_disc_loc_a, temp_disc_dist_a,  temp_disc_nn_loc_a] = discord_discovery_gemm3(a, lengths(lengthIndex), r, display_meta_info);            
        
        if ~isempty(temp_disc_dist_a)
            break;
        else
            if display_meta_info
                r_prev = r;
            end
            r = r*(lengths(lengthIndex)-1)/lengths(lengthIndex);
            if display_meta_info
                fprintf('The suggested value of r = %.2f was too large. Trying again with %.2f\n', r_prev, r);
            end
            disc_misses(lengthIndex) = disc_misses(lengthIndex) + 1;
        end
    end
    
    if length(temp_disc_loc_a) > 1
        [temp_disc_dist_a, top_disc_pos] = max(temp_disc_dist_a);
        temp_disc_loc_a = temp_disc_loc_a(top_disc_pos);
        temp_disc_nn_loc_a = temp_disc_nn_loc_a(top_disc_pos);
    end
    disc_dist_a(lengthIndex) = temp_disc_dist_a;
    disc_loc_a(lengthIndex) = temp_disc_loc_a;
    disc_nn_loc_a(lengthIndex) = temp_disc_nn_loc_a;
    r = temp_disc_dist_a;
    disc_times(lengthIndex) = toc;
end

%PLOT
figure();
subplot(3,1,1);
plot(zscore(a,1));
xlim([1,length(a)]);
box off;
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])

subplot(3,1,2);

temp = zeros(maxLength-minLength+1, size(a,2));
lengths = minLength:maxLength;
for i = 1:length(lengths)
   temp(lengths(i),disc_loc_a(i)) = 1;
end
rectangle('Position',[1,minLength,length(a), maxLength], 'EdgeColor', [0,0,0], 'FaceColor',[0.9,0.9,0.9]);
hold on;
spy(temp,'r');

hold off;
xlim([1,length(a)]);

ylim([minLength, maxLength]);
yticks([minLength, maxLength]);

set(gca,'TickDir','out');
box off;


subplot(3,1,3);

lengths = minLength:maxLength;

hold on;
rectangle('Position',[1,minLength,length(a), maxLength], 'EdgeColor', [0,0,0], 'FaceColor',[0.9,0.9,0.9]);
for i=1:length(lengths)
   plot([disc_loc_a(i),disc_loc_a(i)+lengths(i)],[lengths(i),lengths(i)],'r')
end
hold off;
xlim([1,length(a)]);

ylim([minLength, maxLength]);
yticks([minLength, maxLength]);

set(gca,'TickDir','out');
set(gca, 'YDir','reverse')
box off;
end
