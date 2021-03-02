% --- create transition matrices for each action, p(h'|h,a)

nq = 10; % number of belief states per box
n = nq;

% we need five different transition matrices, one for each of the following actions:
a0=1;   % a0 = action 1 Do nothing
g0=2;   % g0 = action 2 Push button

na = 2; % number of actions
ThA = zeros(n,n,na);

% transition parameters
gamma = .1;   % reward becomes available in box 1
epsilon = .1; % available food disappears from box 1
eta = .001;     % random diffusion of belief

% State rewards
Reward = 1; % reward per time step with food in mouth

%Action costs
pushButtonCost = .5;



%initialize probability distribution over states (belief and world)
pb0 = [1,zeros(1,nq-1)]; % initial belief states (here, lowest availability)

ph0 = pb0;

% setup single-variable transition matrices
Tb = beliefTransitionMatrix(gamma, epsilon, nq, eta);    % default dynamics of beliefs

% ACTION: do nothing
ThA(:,:,1) = Tb; % kronecker product of these transition matrices
ThA(:,:,2) = [zeros(nq-1,nq); ones(1,nq)]; % transition to known state

% ACTION: push button
%bL = ( 1/2 : nq-1/2 ) / nq; % average belief in each bin


Reward_h = Reward*((1:nq)-nq/2); %zeros(1,nq); % state reward
Reward_a = -[0 pushButtonCost]; % rewards are -costs

[R1,R2,R3]=ndgrid(Reward_h,Reward_h,Reward_a);
R = R2 + R3;

for j=1:na,
    ThA(:,:,j) = ThA(:,:,j)'; % convention is p(s,s',a)
end

