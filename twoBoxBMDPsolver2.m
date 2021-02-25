twoboxtask_init;                                % set parameters and define transition matrices
addpath(genpath(pwd));

% Actions:
% { 1=nothing, 2=go 0, 3=go 1, 4=go 2, 5=push
%                       (go actions only go via location 0)

discount = .99; % temporal discount for infinite horizon
niterations = 10000000;

%set Reward matrix over states (belief and world)
[policy, average_reward, cpu_time] = mdp_relative_value_iteration(ThA,R,0.01,10000);
[Q, V, policy, mean_discrepancy] = mdp_Q_learning(ThA,R,discount,niterations);
Qht = reshape(Q,[nq,nr,nq,nl,na]); % joint value of state and action. Format: b2 * r * b1 * l * a

%sc=.3; % color scale
sc = maxx(Qht) - minn(Qht);

subplot(2,2,1);
DelQ1 = squeeze(max(Qht(:,1,:,1,[1 2]),[],5) - max(Qht(:,1,:,1,[3 4]),[],5));
imagesc(bL,bL,DelQ1,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 0, no reward.','color = R(Stay) - R(Go)'});

subplot(2,2,2);
DelQ2 = squeeze(max(Qht(:,1,:,1,[3]),[],5) - max(Qht(:,1,:,1,[4]),[],5));
imagesc(bL,bL,DelQ2,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 0, no reward.','color = R(Go1) - R(Go2)'});

subplot(2,2,3);
DelQ3 = squeeze(max(Qht(:,1,:,2,[1 3]),[],5) - max(Qht(:,1,:,2,[2 4]),[],5));
imagesc(bL,bL,DelQ3,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 1, no reward.','color = R(Stay) - R(Go)'});

subplot(2,2,4);
DelQ4 = squeeze(max(Qht(:,1,:,3,[1 4]),[],5) - max(Qht(:,1,:,3,[2 3]),[],5));
imagesc(bL,bL,DelQ4,sc*[-1 1]); axis xy; axis square; colorbar;
xlabel('belief(food at 1)'); ylabel('belief(food at 2)');
title({'At location 2, no reward.','color = R(Stay) - R(Go)'});


