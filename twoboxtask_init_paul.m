% --- create transition matrices for each action, p(h'|h,a)

% we need five different transition matrices, one for each of the following actions:
a0=1;   % a0 = do nothing
g0=2;   % g0 = go to location 0
g1=3;   % g1 = go toward box 1 (via location 0 if from 2)
g2=4;   % g2 = go toward box 2 (via location 0 if from 1)
pb=5;   % pb  = push button
ThA = cell(1,5);    % transition matrix for each action

% transition parameters
beta = .1;      % available food dropped back into box after button press
gamma1 = .05;   % reward becomes available in box 1
gamma2 = .03;   % reward becomes available in box 2
delta = .01;     % animal trips, doesn't go to target location
epsilon1 = .001; % available food disappears from box 1
epsilon2 = .001; % available food disappears from box 2
rho = .98;       % food in mouth is consumed
eta = .02;      % random diffusion of belief

% Hybrid State h:
%  location
%  belief in food availability at box 1 (nq bins)
%  food in mouth
%  belief in food availability at box 2 (nq bins)

nq = 10; % number of belief states per box
nr = 2; % number of reward states
nl = 3; % number of location states

%initialize probability distribution over states (belief and world)
pr0 = [1 0];  % (r=0, r=1) initially no food in mouth p(R=0)=1.
pl0 = [1 0 0]; % (l=0, l=1, l=2) initial location is at L=0
pb10 = [zeros(1,nq-1),1]; % initial belief states (here, highest availability)
pb20 = [1,zeros(1,nq-1)]; % initial belief states (here, lowest availability)

ph0 = kronn(pl0,pb10,pr0,pb20)'; % kronecker product of these initial distributions
                                 % Note that this ordering makes the subsequent products easiest

% setup single-variable transition matrices
Tr = [1 rho ; 0 1-rho]; % consume reward
Tb1 = beliefTransitionMatrix(gamma1, epsilon1, nq, eta);    % default dynamics of beliefs
Tb2 = beliefTransitionMatrix(gamma2, epsilon2, nq, eta);


% ACTION: do nothing
ThA{1} = kronn(eye(nl),Tb1,Tr,Tb2); % kronecker product of these transition matrices

% ACTION: go to location 0/1/2
Tl0 = [1 1-delta 1-delta ; 0 delta 0 ; 0 0 delta]; % go to loc 0 (with error of delta)
Tl1 = [delta 0 1-delta ; 1-delta 1 0 ; 0 0 delta]; % go to box 1 (with error of delta)
Tl2 = [delta 1-delta 0 ; 0 delta 0 ; 1-delta 0 1]; % go to box 2 (with error of delta)
ThA{2} = kronn(Tl0,Tb1,Tr,Tb2); % go to loc 0
ThA{3} = kronn(Tl1,Tb1,Tr,Tb2); % go to loc 1
ThA{4} = kronn(Tl2,Tb1,Tr,Tb2); % go to loc 2

% ACTION: push button
bL = ( 1/2 : nq-1/2 ) / nq; % average belief in each bin
Trb2 = [ [zeros(nq-1,nq); beta*bL] + diag(1-bL), ...                        % no reward because fail (p=beta), or reward not available
                                                 zeros(nq) ; ...            % food doesn't disappear from mouth here (that's elsewhere)
         [(1-beta)*bL ; zeros(nq-1,nq)], ...                                % yes reward if no fail (1-beta) and reward available; belief is now minimal
                                      diag(1-bL)+[zeros(nq-1,nq); bL] ];   % cannot transfer food, but can observe it
Tb1r = reversekron(Trb2,[2 nq]); % same pattern, but put transitions in correct order by reversing the kronecker product
Th = blkdiag(eye(nq*nr*nq),... % no action if button pressed in location 0
             kron(Tb1r,eye(nq)), ... % try to take reward from box 1, leave box 2 alone
             kron(eye(nq),Trb2));    % try to take reward from box 2, leave box 1 alone
ThA{5} = ThA{1} * Th;   % push the button then wait the usual time step

% State rewards
Reward = 10.; % reward per time step with food in mouth
Groom = 0.8; % location 0 reward
Reward_h = kronn([Groom 0 0],zeros(1,nq),[0 Reward],zeros(1,nq)); % state reward
%[r1,r2,r3,r4] = ndgrid([0 Groom 0],zeros(1,nq),[0 Reward],zeros(1,nq));
[r1,r2,r3,r4] = ndgrid([Groom 0 0],zeros(1,nq),[0 Reward],zeros(1,nq));
r11 = permute(r1,[4 3 2 1]);
r31 = permute(r3,[4 3 2 1]);
Reward_h = r11(:)+r31(:);

%Action costs
travelCost = 0.6;
pushButtonCost = .05;
Reward_a = -[0 travelCost travelCost travelCost pushButtonCost]; % rewards are -costs
[R1,R2,R3]=ndgrid(Reward_h,Reward_h,Reward_a);
R = R2 + R3;

for j=1:5,
    Tb(:,:,j) = ThA{j}';
end

