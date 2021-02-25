% --- create transition matrices for each action, p(h'|h,a)

% Hybrid State h:
%  location
%  belief in food availability at box 1 (nq bins)
%  food in mouth
%  belief in food availability at box 2 (nq bins)

nq = 8; % number of belief states per box
nr = 2; % number of reward states
nl = 3; % number of location states
n = nq*nq*nr*nl; % total number of states


% we need five different transition matrices, one for each of the following actions:
a0=1;   % a0 = do nothing
g0=2;   % g0 = go to location 0
g1=3;   % g1 = go toward box 1 (via location 0 if from 2)
g2=4;   % g2 = go toward box 2 (via location 0 if from 1)
pb=5;   % pb  = push button

na = 5; % number of actions
ThA = zeros(n,n,na);

% transition parameters
beta = .001; %%%%%%%%.001;      % available food dropped back into box after button press
gamma1 = .1;   % reward becomes available in box 1
gamma2 = .1;   % reward becomes available in box 2
delta = .001; %%%%%%%.001;     % animal trips, doesn't go to target location
direct = .001;  %%%%%%%.001;    % animal goes right to target, skipping location 0
epsilon1 = .01; % available food disappears from box 1
epsilon2 = .01; % available food disappears from box 2
rho = .999; %%%%%.999;       % food in mouth is consumed
eta = .0001; %%%%%%% .0001;      % random diffusion of belief

% State rewards
Reward = 3; %3; 
%%%%%%%%%%%%%%%%%%%%%%%% 1;  % reward per time step with food in mouth
Groom = 0.2; %0.2; 
%%%%%%%%%%%%%%%%%%%%%%%% 0.; % location 0 reward

%Action costs
travelCost = .2; %%%%%%%%%% .2;
pushButtonCost = .2; %%%%%%%.2;


%initialize probability distribution over states (belief and world)
pr0 = [1 0];  % (r=0, r=1) initially no food in mouth p(R=0)=1.
pl0 = [1 0 0]; % (l=0, l=1, l=2) initial location is at L=0
pb10 = [1, zeros(1,nq-1)]; % initial belief states (here, lowest availability)
pb20 = [1, zeros(1,nq-1)]; % initial belief states (here, lowest availability)

ph0 = kronn(pl0,pb10,pr0,pb20)'; % kronecker product of these initial distributions
                                 % Note that this ordering makes the subsequent products easiest

% setup single-variable transition matrices
Tr = [1 rho ; 0 1-rho]; % consume reward
Tb1 = beliefTransitionMatrix(gamma1, epsilon1, nq, eta);    % default dynamics of beliefs
Tb2 = beliefTransitionMatrix(gamma2, epsilon2, nq, eta);


% ACTION: do nothing
ThA(:,:,1) = kronn(eye(nl),Tb1,Tr,Tb2); % kronecker product of these transition matrices

% ACTION: go to location 0/1/2
Tl0 = [1 1-delta 1-delta ; 0 delta 0 ; 0 0 delta]; % go to loc 0 (with error of delta)
Tl1 = [delta 0 1-delta-direct ; 1-delta 1 direct ; 0 0 delta]; % go to box 1 (with error of delta)
Tl2 = [delta 1-delta-direct 0 ; 0 delta 0 ; 1-delta direct 1]; % go to box 2 (with error of delta)
%Tl1 = [delta 0 1-delta ; 1-delta 1 0 ; 0 0 delta]; % go to box 1 (with error of delta)
%Tl2 = [delta 1-delta 0 ; 0 delta 0 ; 1-delta 0 1]; % go to box 2 (with error of delta)
ThA(:,:,2) = kronn(Tl0,Tb1,Tr,Tb2); % go to loc 0
ThA(:,:,3) = kronn(Tl1,Tb1,Tr,Tb2); % go to loc 1
ThA(:,:,4) = kronn(Tl2,Tb1,Tr,Tb2); % go to loc 2

% ACTION: push button
bL = ( 1/2 : nq-1/2 ) / nq; % average belief in each bin
%Trb2 = [ [zeros(nq-1,nq); beta*bL] + diag(1-bL), ...                        % no reward because fail (p=beta), or reward not available
%                                                 zeros(nq) ; ...            % food doesn't disappear from mouth here (that's elsewhere)
%         [(1-beta)*bL ; zeros(nq-1,nq)], ...                                % yes reward if no fail (1-beta) and reward available; belief is now minimal
%                                      diag(1-bL)+[zeros(nq-1,nq) ; bL] ];   % cannot transfer food, but can observe it

Trb2 = [ 1-bL zeros(1,nq) ; ...                     % sure there's no food 1-b of the time
         zeros(nq-2,2*nq) ; ...
         beta*bL zeros(1,nq) ; ...
         ...
         (1-beta)*bL 1-bL ; ...
         zeros(nq-2,2*nq); ...
         zeros(1,nq) bL]; % transition probability from belief state b to b' given Action = LOOK

Tb1r = reversekron(Trb2,[2 nq]); % same pattern, but put transitions in correct order by reversing the kronecker product
Th = blkdiag(eye(nq*nr*nq),... % no action if button pressed in location 0
             kron(Tb1r,eye(nq)), ... % try to take reward from box 1, leave box 2 alone
             kron(eye(nq),Trb2));    % try to take reward from box 2, leave box 1 alone
ThA(:,:,5) = Th * ThA(:,:,1) ;
%%%%%%%%%Th * ThA(:,:,1);   % wait the usual time step then push the button

Reward_h = tensorsumm([Groom 0 0],zeros(1,nq),[0 Reward],zeros(1,nq)); % state reward
%Reward_h = tensorsumm([0 Groom 0],zeros(1,nq),[0 Reward],zeros(1,nq)); % state reward

Reward_a = -[0 travelCost travelCost travelCost pushButtonCost]; % rewards are -costs

[R1,R2,R3]=ndgrid(Reward_h,Reward_h,Reward_a);
R = R2 + R3;

for j=1:5,
    ThA(:,:,j) = ThA(:,:,j)'; % convention is p(s,s',a)
end

