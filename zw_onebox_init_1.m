% --- create transition matrices for each action, p(h'|h,a)
clear;

nq = 5; % number of belief states per box
nr = 2;
n = nq * nr;

% we need two different transition matrices, one for each of the following actions:
a0=1;   % a0 = action 1 Do nothing
g0=2;   % g0 = action 2 Push button

na = 2; % number of actions
ThA = zeros(n,n,na);

% transition parameters
gamma = .1;   % reward becomes available in box 1
epsilon = .01; % available food disappears from box 1
eta = .00001;     % random diffusion of belief

% State rewards
Reward = 1; % reward per time step with food in mouth

%Action costs
pushButtonCost = .5; %.5; %%%%%%%%.5;


%initialize probability distribution over states (belief and world)
pb0 = [1,zeros(1,nq-1)]; % initial belief states (here, lowest availability)
pr0 = [1 0];  % (r=0, r=1) initially no food in mouth p(R=0)=1.

ph0 = kronn(pr0, pb0)'; 

% setup single-variable transition matrices
rho = 0.99;
Tr = [1 rho ; 0 1-rho]; % consume reward
Tb = beliefTransitionMatrix(gamma, epsilon, nq, eta);    % default dynamics of beliefs

% ACTION: do nothing
ThA(:,:,1) = kronn(Tr, Tb); % the same as the default dynamics of beliefs
%ThA(:,:,2) = [zeros(nq-1,nq); ones(1,nq)]; % transition to known state

% ACTION: push button
bL = ( 1/2 : nq-1/2 ) / nq; % average belief in each bin
Trb = [ 1-bL zeros(1,nq) ; ...                     % sure there's no food 1-b of the time
    zeros(nq-2,2*nq) ; ...
    zeros(1,nq) zeros(1,nq) ; ...
    ...
    bL 1-bL ; ...
    zeros(nq-2,2*nq); ...
    zeros(1,nq) bL]; % transition probability from belief state b to b' given Action = LOOK
%ThA(:,:,2) = reversekron(Trb,[2 nq]);
ThA(:,:,2) = Trb * ThA(:, :, 1);

Reward_h = tensorsumm([0 Reward],zeros(1,nq)); % state reward
Reward_a = -[0 pushButtonCost]; % rewards are -costs

[R1,R2,R3]=ndgrid(Reward_h,Reward_h,Reward_a);
R = R2 + R3;
%R(:, :, 1) = Reward_h';
%R(:, :, 2) = Reward_h' - pushButtonCost;

for j=1:na,
    ThA(:,:,j) = ThA(:,:,j)'; % convention is p(s,s',a)
end

