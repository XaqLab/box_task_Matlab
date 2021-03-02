%The 2 option VI schedule MDP problem
% We give ourselves 6 states. We have the following
% Reward availability site 1 in {0, 1}
% Reward availability site 2 in {0, 1}
% these are not mutually exclusive, but the pairs 0,0; 1,0; 0,1 and 1,1 are
% In addition, we have reward delivered at site 1 and at site 2, which we
% treat as separate states, entered via actions, and effecting transitions
% as follows.
% reward delivered at site 1 and 2 are mutually exclusive, and
% reward delivered at site 1 (rd1) is mutually exclusive with  1,0; and 1,1,
% effecting the transition rd1 => [ 1,0 ] -> [0,0]; and [1,1] -> [0,1]
% while reward delivered at site 2 (rd2) is mutually exclusive with  0,1; and 1,1,
% effecting the transition rd2 => [ 0,1 ] -> [0,0]; and [1,1] -> [1,0]
%  the state s has 4 values
% state 1: 0,0  (reward at neither side)
% state 2: 1,0  (reward at site 1 only)
% state 3: 0,1  (reward at site 2 only)
% state 4: 1,1  (reward at both sites)
% we can concatenate a binary state to track the last action, to incorporate switching
% costs, if needed, for an 8 state vector.
%
% We also have three actions, a \in { wait, select 1, and select 2 }.
%   The reward function is R(s,a) is 4x3, with costs for actions 2 and 3,
%   no cost for action 1
% rd1  (reward delivered site 1) is the reward for site 1 
% rd2  (reward delivered site 2) is the reward for site 2 
% assuming these are equal and set to one, and action costs are 0 -c 
%  R(s,a) is 
%   null a1 a2
c = 0.2;
R = [ 0 -c -c;   % 00
      0 1-c -c;  % 10
      0 -c  1-c; % 01
      0 1-c 1-c ]; % 11
      
%  We have a probabilistic replishment schedule, where the probability of
%  availability is zero after delivered, which grows according to it's rate
%  schedule.  
% With these ideas in place, we can define a state dependent transition
% matrix Tn = p(s'|s,a)
% Schedules are defined by rates R like 25 or 15 which indicate the expected
% number of time steps before available, which means 1/R is the transition
% probability from unavailable to available.
p1 = 1/25;
p2 = 1/15;
% Under the NULL action
%  00 -> 00 with probability 1-[ p1(1-p2) +  p2(1-p1) + p1*p2]
%  00 -> 10 with probability p1(1-p2) 
%  00 -> 01 with probability  p2(1-p1)
%  00 -> 11 with probability p1*p2
%%%%
%  01 -> 11 with probability p2 
%  01 -> 01 with probability 1-p2
%  10 -> 11 with probability p1
%  10 -> 10 with probability 1-p1
%  11 -> 11 with probability 1
% else 0
Tn(:,:,1) = [1-(p1*(1-p2)+p2*(1-p1)+p1*p2)    0    0     0;
                     p1*(1-p2)              1-p1   0     0;
                     p2*(1-p1)               0    1-p2   0;
                      p1*p2                  p1    p2    1 ];

% Under the select 1 action
%  10 -> 00 with probability 1
%  10 -> 10 with probability 0
%  11 -> 01 with probability 1
%  11 -> 11 with probability 0
% else self transition (diagonal)
Tn(:,:,2) = [ 1 1 0 0;
              0 0 0 0;
              0 0 1 1;
              0 0 0 0 ];

% Under the select 2 action
%  01 -> 00 with probability 1
%  01 -> 01 with probability 0
%  11 -> 10 with probability 1
%  11 -> 11 with probability 0
% else self transition (diagonal)

Tn(:,:,3) = [ 1 0 1 0;
              0 1 0 1;
              0 0 0 0;
              0 0 0 0 ];
Tn = permute(Tn,[ 2 1 3]);
[policy, average_reward, cpu_time] = mdp_relative_value_iteration(Tn,R,0.01,1000);

% policy =
% 
%      1
%      2
%      3
%      2
% 
% 
% average_reward =
% 
%     0.0767

% can we do better by using the probalities of p1, p2 on a grid as the
% state?
% here I write down a BELIEF MDP.  My state is [p1(t);p2(t)].  the transition
% dynamics are a continuous time MDP.  Writing the diff EQ for the four
% states we have 
%  [p1(t+dt);p2(t+dt)] = [     ]*[p1(t);p2(t)]
% sample these 2D belief probabilities and vectorize  
% Now I need a transition matrix that acts to carry a probability on
% on the belief vector into an updated vector.  Initialize the vector.
% the natural dynamics is a shift of the probabilites according to 

