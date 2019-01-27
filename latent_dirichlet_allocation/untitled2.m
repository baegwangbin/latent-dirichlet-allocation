% Latent Dirichlet Allocation applied to the KOS dataset
% ADVICE: consider doing clear, close all
clear all
clc

load kos_doc_data.mat

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 0.1;    % ORIGINAL parameter of the Dirichlet over topics for one document
%alpha = 10;
gamma = 0.1;    % parameter of the Dirichlet over words

% A's columns are doc_id, word_id, count
swd = sparse(A(:,2),A(:,1),A(:,3));
Swd = sparse(B(:,2),B(:,1),B(:,3));

perplexity_iter = zeros(length(K),1);
perplexity_K = zeros(length(K), 1);

%loop for the number of topics
for t = 1:length(K)

% Initialization: assign each word in each document a topic
skd = zeros(K(t),D); % count of word assignments to topics for document d
swk = zeros(W,K(t)); % unique word topic assignment counts accross all documents
s = cell(D, 1);   % one cell for every document
for d = 1:D                % cycle through the documents
  z = zeros(W,K(t));          % unique word topic assignment counts for doc d
  for w = A(A(:,1)==d,2)'  % loop over unique words present in document d
    c = swd(w,d);          % number of occurences of word w in document d
    for i=1:c    % assign each occurence of word w to a topic at random
      k = ceil(K(t)*rand());
      z(w,k) = z(w,k) + 1;
    end
  end
  skd(:,d) = sum(z,1)';  % number of words in doc d assigned to each topic
  swk = swk + z;  % unique word topic assignment counts accross all documents
  s{d} = sparse(z); % sparse representation: z contains many zero entries
end
sk = sum(skd,2);  % word to topic assignment counts accross all documents

iterations = 50;
theta = zeros(K(t), size(iterations,2));
beta = zeros(1, K(t));
num_topic_perplexity = zeros(length(iterations), length(K));
word_entropy = zeros(K(t),iterations); %word entropy for all topics
beta_fixed_word_sweeps = zeros(K(t), iterations);
    
    
% This makes a number of Gibbs sampling sweeps through all docs and words
for iter = 1:iterations    % This can take a couple of minutes to run
    
  for d = 1:D
    z = full(s{d}); % unique word topic assigmnet counts for document d
    for w = A(A(:,1)==d,2)' % loop over unique words present in document d
      a = z(w,:); % number of times word w is assigned to each topic in doc d
      ka = find(a); % topics with non-zero counts for word w in document d
      for k = ka(randperm(length(ka))) % loop over topics in permuted order
        for i = 1:a(k) % loop over counts for topic k
          z(w,k) = z(w,k) - 1;      % remove word from count matrices
          swk(w,k) = swk(w,k) - 1;
          sk(k)    = sk(k)    - 1;
          skd(k,d) = skd(k,d) - 1;
          b = (alpha + skd(:,d)) .* (gamma + swk(w,:)') ./ (W*gamma + sk);
          kk = sampDiscrete(b);     % Gibbs sample new topic assignment
          z(w,kk) = z(w,kk) + 1;    % add word with new topic to count matrices
          skd(kk,d) = skd(kk,d) + 1;
          swk(w,kk) = swk(w,kk) + 1;
          sk(kk) =    sk(kk)    + 1;
        end
      end
    end
    s{d} = sparse(z);   % store back into sparse structure
  end
  
     %theta for each document
    theta(:,iter) = ( skd(:,2) + alpha ) / (sum( skd(:,2) + alpha)) ; 
 
    beta = zeros(W,K(t));
    for o = 1:K(t)
        beta(:,o) = ( swk(:,o) + gamma ) / (sum( swk(:,o) + gamma)) ; 
    end
    
    %WORD 10
    beta_fixed_word = beta(10,:)';
    beta_fixed_word_sweeps(:, iter) = beta_fixed_word;
    
    for i = 1:K(t)
    entropy = 0;   %word entropy for topic
        for j = 1:size(beta,1)
            entropy = entropy + (-beta(j,i)*log(beta(j,i)));
        end
    word_entropy(i,iter) = entropy;
    end    
end
%plotting theta for each document
%plotting topic posteriors
figure(1)
plot(theta);

figure(2)
plot(beta);

figure(3)
plot(word_entropy')

% compute the perplexity for all words in the test set B
% We need the new Skd matrix, derived from corpus B
lp = 0; nd = 0;
for d = unique(B(:,1))'  % loop over all documents in B
  % randomly assign topics to each word in test document d
  z = zeros(W,K(t));
  for w = B(B(:,1)==d,2)'   % w are the words in doc d
    for i=1:Swd(w,d)
      k = ceil(K(t)*rand());
      z(w,k) = z(w,k) + 1;
    end
  end
  Skd = sum(z,1)';
  Sk = sk + Skd;  
  
   
  % perform some iterations of Gibbs sampling for test document d
  for iter = 1:iterations
    for w = B(B(:,1)==d,2)' % w are the words in doc d
      a = z(w,:); % number of times word w is assigned to each topic in doc d
      ka = find(a); % topics with non-zero counts for word d in document d
      for k = ka(randperm(length(ka)))
        for i = 1:a(k)
          z(w,k) = z(w,k) - 1;   % remove word from count matrix for doc d
          Skd(k) = Skd(k) - 1;
          b = (alpha + Skd) .* (gamma + swk(w,:)') ./ (W*gamma + sk);
          kk = sampDiscrete(b);
          z(w,kk) = z(w,kk) + 1; % add word with new topic to count matrix for doc d
          Skd(kk) = Skd(kk) + 1;
        end
      end
    end
  end
  b=(alpha+Skd')/sum(alpha+Skd)*bsxfun(@rdivide,gamma+swk',W*gamma+sk);  
  w=B(B(:,1)==d,2:3);
  lp = lp + log(b(w(:,1)))*w(:,2);   % log probability, doc d
  nd = nd + sum(w(:,2));             % number of words, doc d
end
%computing perplexity for documents B
perplexity = exp(-lp/nd);   % perplexity

%relation between perplexity and number of iterations
perplexity_K(t,1) = perplexity;




end

figure(4)
plot(perplexity_iter);


% this code allows looking at top I words for each mixture component
I = 20;
for k=1:K, [i ii] = sort(-swk(:,k)); ZZ(k,:)=ii(1:I); end
for i=1:I, for k=1:K, fprintf('%-15s',V{ZZ(k,i)}); end; fprintf('\n'); end

% test = []
% for i = 1:2000
% if sd(i,1)==15
% test = [test;i];
% end
% end


%explanation of plots

%EXTENSIONS
%CHANGE THE VALUES OF ALPHA AND BETA AND OBSERVE RESULTS
% MODEL SELECTION (SEE GRIFFITHS PAPER)