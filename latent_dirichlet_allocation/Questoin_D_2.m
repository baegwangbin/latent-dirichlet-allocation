% Answer to Question D
% Changing random seeds

clear all
clc
load kos_doc_data.mat

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 10;                 % parameter of the Dirichlet over mixture components
gamma = 0.1;                % parameter of the Dirichlet over words

theta_seed_1 = zeros(20,100);
theta_seed_2 = zeros(20,100);
theta_seed_3 = zeros(20,100);

for seed = 1:3

    rng('default');
    rng(seed);

    % Initialization: assign each document a mixture component at random
    sd = ceil(K*rand(D,1));     % mixture component assignment of each document
    swk = zeros(W,K);           % K multinomials over W unique words
    sk_docs = zeros(K,1);       % number of documents assigned to each mixture component

    % This populates count matrices swk, sk_docs and sk_words
    for d = 1:D                % cycle through the documents
      w = A(A(:,1)==d,2);      % unique words in doc d
      c = A(A(:,1)==d,3);      % counts
      k = sd(d);               % doc d is in mixture k
      swk(w,k) = swk(w,k) + c; % num times word w is assigned to mixture component k
      sk_docs(k) = sk_docs(k) + 1;
    end
    sk_words = sum(swk,1)';    % num words assigned to mixture component k accross all docs


    theta = zeros(20,100);

    % This makes a number of Gibbs sampling sweeps through all docs and words
    for iter = 1:100     % number of Gibbs sweeps
      for d = 1:D       % for each document iterate through all its words
        w = A(A(:,1)==d,2);    % unique words in doc d
        c = A(A(:,1)==d,3);    % counts

        swk(w,sd(d)) = swk(w,sd(d)) - c;  % remove doc d words from count table
        sk_docs(sd(d)) = sk_docs(sd(d)) - 1;        % remove document counts
        sk_words(sd(d)) = sk_words(sd(d)) - sum(c); % remove total word counts

        lb = zeros(1,K);    % log probability of doc d under mixture component k
        for k = 1:K
          ll = c'*( log(swk(w,k)+gamma) - log(sk_words(k) + gamma*W) );
          lb(k) = log(sk_docs(k) + alpha) + ll;
        end

        b = exp(lb-max(lb));  % exponentiation of log probability plus constant
        kk = sampDiscrete(b); % sample from unnormalized discrete distribution

        swk(w,kk) = swk(w,kk) + c;        % add back document word counts
        sk_docs(kk) = sk_docs(kk) + 1;              % add back document counts
        sk_words(kk) = sk_words(kk) + sum(c);       % add back document counts
        sd(d) = kk;    
      end

      theta(:,iter) = (sk_docs(:) + alpha) / sum(sk_docs(:) + alpha);

    end
    
    if seed == 1
        theta_seed_1 = theta;
    elseif seed == 2
        theta_seed_2 = theta;
    elseif seed == 3
        theta_seed_3 = theta;
    end

end

subplot(1,3,1);
plot(theta_seed_1', 'Linewidth', 1)
grid on;
set(gca,'fontsize',13);
title('Seed = 1', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Posterior Probabilities', 'FontSize', 15, 'FontWeight', 'bold');
xlim([1,100])
ylim([0,0.35])

subplot(1,3,2); 
plot(theta_seed_2', 'Linewidth', 1)
grid on;
set(gca,'fontsize',13);
title('Seed = 2', 'FontSize', 20, 'FontWeight', 'bold');
xlabel('Gibbs Iteration', 'FontSize', 15, 'FontWeight', 'bold');
xlim([1,100])
ylim([0,0.35])

subplot(1,3,3);
plot(theta_seed_3', 'Linewidth', 1)
grid on;
set(gca,'fontsize',13);
title('Seed = 3', 'FontSize', 20, 'FontWeight', 'bold');
xlim([1,100])
ylim([0,0.35])