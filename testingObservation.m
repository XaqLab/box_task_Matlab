% evolution of belief and reward time distribution under different
% observation sequences

% assuming discrete time

T = 100; % number of time points
gamma = .03; % probability of food becoming available in one time step, given that it was not available before
epsilon = .01; % probability of food becoming UNavailable in one time step, given that it WAS available before
nq = 400; % number of distinct belief states
w = .0000001; % stochasticity on belief space
bL = (.5:nq-.5)/nq; % typical beliefs in each bin
Tb = beliefTransitionMatrix(gamma,epsilon,nq,w); % in absence of observations

% belief space only
TrbA = zeros(nq,nq,2); % Transition matrices for two possible actions
TrbA(:,:,1) = Tb; % transition probability from belief state b to b' given Action = WAIT
TrbA(:,:,2) = TrbA(:,:,1)*[ bL; zeros(nq-2,nq); 1-bL]; % transition probability from belief state b to b' given Action = LOOK


b0=[1; zeros(nq-1,1)]; % initial belief in availability states being 0,1
bL = zeros(nq,T); % storage for belief dynamics
aL = 1+(rand(1,T)>.9); % actions


bL(:,1) = b0;
for t=2:T
    bL(:,t) = TrbA(:,:,aL(t)) * bL(:,t-1);
end
