twoboxtask_init;                                % set parameters and define transition matrices
addpath(genpath(pwd));

% we need five different transition matrices, one for each of the following actions:
% a0=1;   % a0 = do nothing
% g0=2;   % g0 = go to location 0
% g1=3;   % g1 = go toward box 1 (via location 0 if from 2)
% g2=4;   % g2 = go toward box 2 (via location 0 if from 1)
% pb=5;   % pb  = push button

%set Reward matrix over states (belief and world)
[policy, average_reward, cpu_time] = mdp_relative_value_iteration(Tb,R,0.01,10000);
[Q, V, policy, mean_discrepancy] = mdp_Q_learning(Tb,R,0.001,500000);
na = 5;
Qht = reshape(Q,[nq,nr,nq,nl,na]); % joint distribution. Format: b2 * r * b1 * l
Qrt(:,:) = squeeze(sum(sum(sum(Qht(:,:,:,:,:),1),3),4)); % marginal over reward
Qlt(:,:) = squeeze(sum(sum(sum(Qht(:,:,:,:,:),1),2),3)); % marginal over location
Qb1t(:,:) = squeeze(sum(sum(sum(Qht(:,:,:,:,:),1),2),4)); % marginal over beliefs b1
Qb2t(:,:) = squeeze(sum(sum(sum(Qht(:,:,:,:,:),2),3),4)); % marginal over beliefs b2
Qb1b2t(:,:,:) = squeeze(sum(sum(Qht(:,:,:,:,:),2),4)); % bivariate marginal over beliefs b1,b2

subplot(3,1,1);
DelQ1 = squeeze(max(Qht(:,1,:,1,[1 2]),[],5) - max(Qht(:,1,:,1,3:4),[],5));
imagesc(0.1:0.1:1,0.1:0.1:1,exp(DelQ1)); axis xy; axis square; colorbar; figure(gcf)
title('At location 0, no reward, Better to stay');

subplot(3,1,2);
DelQ2 = squeeze(max(Qht(:,1,:,2,[2 4]),[],5) - max(Qht(:,1,:,2,[5 3 1]),[],5));
imagesc(0.1:0.1:1,0.1:0.1:1,exp(DelQ2)); axis xy; axis square; colorbar; figure(gcf)
title('At location 1, no reward, Better to switch');

subplot(3,1,3);
DelQ3 = squeeze(max(Qht(:,1,:,3,[2 3]),[],5) - max(Qht(:,1,:,3,[5 4 1]),[],5));
imagesc(0.1:0.1:1,0.1:0.1:1,exp(DelQ3)); axis xy; axis square; colorbar; figure(gcf)
title('At location 2, no reward, Better to switch');


