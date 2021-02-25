% evolution of belief and reward time distribution under different
% observation sequences

% assuming discrete time

T = 200; % number of time points
gamma = .03; % probability of food becoming available in one time step, given that it was not available before
epsilon = .00000001; % probability of food becoming UNavailable in one time step, given that it WAS available before
nq = 40; % number of distinct belief states
w = .0000001; % stochasticity on belief space
bL = (.5:nq-.5)/nq; % typical beliefs in each bin
Tb = beliefTransitionMatrix(gamma,epsilon,nq,w); % in absence of observations

% joint space
TrbA = zeros(2*nq,2*nq,2); % Transition matrices for two possible actions
TrbA_xaq = TrbA;
TrbA_paul = TrbA;
TrbA(:,:,1) = kron([1 1; 0 0],Tb); % transition probability from belief state b to b' given Action = WAIT; food disappears

TrbA_xaq(:,:,1) = TrbA(:,:,1); % we agree under the no-look action
TrbA_xaq(:,:,2) = [ 1-bL 1-bL; zeros(nq-1,2*nq) ; bL bL ; zeros(nq-1,2*nq)]*TrbA(:,:,1); % xaq's transition probability from belief state b to b' given Action = LOOK

TrbA_paul(:,:,1) = TrbA(:,:,1); % we agree under the no-look action
TrbA_paul(:,:,2) = [ diag(1-bL) diag(1-bL) ; bL bL ; zeros(nq-1,2*nq)]*TrbA(:,:,1); % paul's transition probability from belief state b to b' given Action = LOOK

availableRewards = [zeros(nq,1); ones(nq,1)];


b0=[1; zeros(2*nq-1,1)]; % initial belief in availability states being 0,1
bL_xaq = zeros(2*nq,T); % storage for belief dynamics
bL_paul = zeros(2*nq,T); % storage for belief dynamics

aLtot = {...
    1*ones(1,T), ...  % never look
    2*ones(1,T), ...  % always look
    1+(rand(1,T)>.9)}; % look randomly
nexamples = length(aLtot);
actionLabel = {'never look', 'always look', 'look randomly'};

figure;
i=0;
for e=1:nexamples
    i=i+3;
    bL_xaq(:,1) = b0;
    bL_paul(:,1) = b0;
    for t=2:T
        bL_xaq(:,t) = TrbA_xaq(:,:,aLtot{e}(t)) * bL_xaq(:,t-1);
        bL_paul(:,t) = TrbA_paul(:,:,aLtot{e}(t)) * bL_paul(:,t-1);
    end

    subplot(nexamples,3,i-2);
    imagesc(bL_xaq); colorbar; ylabel('{r_t,p(x_t=1|obs)}'); title(['posterior dynamics given T_{hh''}^{xaq} and ',actionLabel{e}]); xlabel('time');

    subplot(nexamples,3,i-1);
    imagesc(bL_paul); colorbar; ylabel('{r_t,p(x_t=1|obs)}'); title(['posterior dynamics given T_{hh''}^{paul} and ',actionLabel{e}]); xlabel('time');

    subplot(nexamples,3,i);
    plot(1:T,cumsum(bL_xaq'*availableRewards),'b', 1:T,cumsum(bL_paul'*availableRewards),'r');
    if e==2
        hold on
        plot(1:T,gamma * (1:T),'g');
    end
    
end
