clear;
close all;
zw_onebox_init_1;                                % set parameters and define transition matrices
addpath(genpath(pwd));

% Actions:
% { 1=nothing, 2=push button }

discount = .99; % temporal discount for infinite horizon
niterations = 10000000;

%set Reward matrix over states (belief and world)
[policy1, average_reward, cpu_time] = mdp_relative_value_iteration(ThA,R,discount,10000);
policy1'

%[Q, V, policy2, mean_discrepancy] = mdp_Q_learning(ThA,R,discount,niterations);
[Q, V, policy2, mean_discrepancy, stateTrajectory] = mdp_Q_learning_modified(ThA,R,discount,niterations);
policy2'

% [V, policy3, iter, cpu_time]= mdp_policy_iteration(ThA,R,discount);
% policy3'
%%
Qht = reshape(Q, [nq, nr, na]); 

figure; 
histogram(stateTrajectory,n);

%figure;
%plot(stateTrajectory, 'b.-');

figure; 
hold on;
p1 = plot(Q(:, 1), 'b');    % Q-value of doing nothing
p2 = plot(Q(:, 2), 'r');         % Q-value of pressing the button
legend([p1, p2], 'doing nothing', 'press button');
%p3 = plot(max(Q, [], 2), 'g');        
%legend([p1, p2, p3], 'doing nothing', 'press button', 'optimal');
hold off;


% sc=.3; % color scale
% %sc = maxx(Qht) - minn(Qht);
% 
% subplot(2,2,1);
% DelQ1 = squeeze(max(Qht(:,1,:,1,[1 2]),[],5) - max(Qht(:,1,:,1,[3 4]),[],5));
% imagesc(bL,bL,DelQ1,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 0, no reward.','color = R(Stay) - R(Go)'});
% 
% subplot(2,2,2);
% DelQ2 = squeeze(max(Qht(:,1,:,1,[3]),[],5) - max(Qht(:,1,:,1,[4]),[],5));
% imagesc(bL,bL,DelQ2,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 0, no reward.','color = R(Go1) - R(Go2)'});
% 
% subplot(2,2,3);
% DelQ3 = squeeze(max(Qht(:,1,:,2,[1 3]),[],5) - max(Qht(:,1,:,2,[2 4]),[],5));
% imagesc(bL,bL,DelQ3,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 1, no reward.','color = R(Stay) - R(Go)'});
% 
% subplot(2,2,4);
% DelQ4 = squeeze(max(Qht(:,1,:,3,[1 4]),[],5) - max(Qht(:,1,:,3,[2 3]),[],5));
% imagesc(bL,bL,DelQ4,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 2, no reward.','color = R(Stay) - R(Go)'});
% 
% 
