%clear; 
zw_twoboxtask_init;                                % set parameters and define transition matrices
addpath(genpath(pwd));

% Actions:
% { 1=nothing, 2=go 0, 3=go 1, 4=go 2, 5=push
%                       (go actions only go via location 0)

discount = 0.99;        % temporal discount for infinite horizon
niterations = 1000000;

%[policy1, average_reward, cpu_time] = mdp_relative_value_iteration(ThA,R);
% The original value-iteration algorithm in the toolbox

[policy1, average_reward, cpu_time, V1, Q1] = mdp_value_iteration_modified(ThA,R, discount);
% modified value-iteration algorithm, value and Q-value are returned

%[Q2, V2, policy2, mean_discrepancy] = mdp_Q_learning(ThA,R,discount,niterations);
% The original value-iteration algorithm in the toolbox

%[Q2, V2, policy2, mean_discrepancy, stateTrajectory, actionTrajectory] = mdp_Q_learning_modified(ThA,R,discount,niterations);
% modified Q-learning algorithm, value and Q-value are returned

%[V3, policy3, cpu_time, Q3] = mdp_LP_modified(ThA,R,discount);


%% value-iteration
Q = Q1;   
Qht = reshape(Q,[nq,nr,nq,nl,na]); % joint value of state and action. Format: b2 * r * b1 * l * a

figure;
sc=.3; % color scale
%sc = maxx(Qht) - minn(Qht);
subplot(2,2,1);
DelQ1 = squeeze(max(Qht(:,1,:,1,[a0 g0]),[],5) - max(Qht(:,1,:,1,[g1 g2]),[],5));
imagesc(bL,bL,DelQ1,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 0, no reward.','color = R(Stay) - R(Go)'});

subplot(2,2,2);
DelQ2 = squeeze(max(Qht(:,1,:,1,[g1]),[],5) - max(Qht(:,1,:,1,[g2]),[],5));
imagesc(bL,bL,DelQ2,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 0, no reward.','color = R(Go1) - R(Go2)'});

subplot(2,2,3);
DelQ3 = squeeze(max(Qht(:,1,:,2,[a0 g1]),[],5) - max(Qht(:,1,:,2,[g0 g2]),[],5));
imagesc(bL,bL,DelQ3,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 1, no reward.','color = R(Stay) - R(Go)'});

subplot(2,2,4);
DelQ4 = squeeze(max(Qht(:,1,:,3,[a0 g2]),[],5) - max(Qht(:,1,:,3,[g0 g1]),[],5));
imagesc(bL,bL,DelQ4,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 2, no reward.','color = R(Stay) - R(Go)'});

% subplot(3,2,5);
% DelQ5 = squeeze(max(Qht(:,2,:,2,[a0 g1 pb]),[],5) - max(Qht(:,2,:,2,[g0 g2]),[],5));
% imagesc(bL,bL,DelQ5,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 1, has reward.','color = R(Stay) - R(Go)'});
% 
% subplot(3,2,6);
% DelQ6 = squeeze(max(Qht(:,2,:,3,[a0 g2 pb]),[],5) - max(Qht(:,2,:,3,[g0 g1]),[],5));
% imagesc(bL,bL,DelQ6,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 2, has reward.','color = R(Stay) - R(Go)'});



%% Q-learning
% Q = Q2;   
% Qht = reshape(Q,[nq,nr,nq,nl,na]); % joint value of state and action. Format: b2 * r * b1 * l * a
% 
% figure; 
% histogram(stateTrajectory,n);
% xlabel('state'); ylabel('times')
% title('Histogram of visiting time of each state');
%  
% figure;
% plot(stateTrajectory, 'b.-');
% xlabel('time'); ylabel('state')
% title('State trajectory during iteration');
%  
% 
% figure;
% sc=.3;
% subplot(2,2,1);
% DelQ1 = squeeze(max(Qht(:,1,:,1,[a0 g0]),[],5) - max(Qht(:,1,:,1,[g1 g2]),[],5));
% imagesc(bL,bL,DelQ1,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 0, no reward.','color = R(Stay) - R(Go)'});
% 
% subplot(2,2,2);
% DelQ2 = squeeze(max(Qht(:,1,:,1,[g1]),[],5) - max(Qht(:,1,:,1,[g2]),[],5));
% imagesc(bL,bL,DelQ2,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 0, no reward.','color = R(Go1) - R(Go2)'});
% 
% subplot(2,2,3);
% DelQ3 = squeeze(max(Qht(:,1,:,2,[a0 g1]),[],5) - max(Qht(:,1,:,2,[g0 g2]),[],5));
% imagesc(bL,bL,DelQ3,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 1, no reward.','color = R(Stay) - R(Go)'});
% 
% subplot(2,2,4);
% DelQ4 = squeeze(max(Qht(:,1,:,3,[a0 g2]),[],5) - max(Qht(:,1,:,3,[g0 g1]),[],5));
% imagesc(bL,bL,DelQ4,sc*[-1 1]); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 2, no reward.','color = R(Stay) - R(Go)'});
% 
% subplot(3,2,5);
% DelQ5 = squeeze(max(Qht(:,2,:,2,[1 3]),[],5) - max(Qht(:,2,:,2,[2 4]),[],5));
% imagesc(bL,bL,exp(DelQ5)); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 1, has reward.','color = R(Stay) - R(Go)'});
% 
% subplot(3,2,6);
% DelQ6 = squeeze(max(Qht(:,2,:,3,[1 4]),[],5) - max(Qht(:,2,:,3,[2 3]),[],5));
% imagesc(bL,bL,exp(DelQ6)); axis xy; axis square; colorbar;
% xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
% title({'At location 2, has reward.','color = R(Stay) - R(Go)'});
 
%% 
% check how many times each state in subfigure 1 has been visited
state = 1:n;
SM = reshape(state, [nq,nr,nq,nl]);
Del1_SM = squeeze(SM(:,1, :,1));   % corresponds to states at location 0, no reward
counts = histcounts(stateTrajectory,n);
counts(Del1_SM)