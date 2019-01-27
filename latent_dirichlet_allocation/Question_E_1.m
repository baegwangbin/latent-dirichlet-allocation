% Answer to Question E
% Topic Posteriors vs Gibbs Iteration
% Word Entropy vs Gibbs Iteration

clear all
clc
load kos_doc_data.mat

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 0.1;    % parameter of the Dirichlet over topics for one document
gamma = 0.1;    % parameter of the Dirichlet over words

% A's columns are doc_id, word_id, count
swd = sparse(A(:,2),A(:,1),A(:,3));
Swd = sparse(B(:,2),B(:,1),B(:,3));

%%%%
%iterations = 20
%perplexity_iter = zeros(20,20)
%%%%

rng('default');
rng(1);

% Initialization: assign each word in each document a topic
skd = zeros(K,D); % count of word assignments to topics for document d
swk = zeros(W,K); % unique word topic assignment counts accross all documents
s = cell(D, 1);   % one cell for every document
for d = 1:D                % cycle through the documents
  z = zeros(W,K);          % unique word topic assignment counts for doc d
  for w = A(A(:,1)==d,2)'  % loop over unique words present in document d
    c = swd(w,d);          % number of occurences of word w in document d
    for i=1:c    % assign each occurence of word w to a topic at random
      k = ceil(K*rand());
      z(w,k) = z(w,k) + 1;
    end
  end
  skd(:,d) = sum(z,1)';  % number of words in doc d assigned to each topic
  swk = swk + z;  % unique word topic assignment counts accross all documents
  s{d} = sparse(z); % sparse representation: z contains many zero entries
end
sk = sum(skd,2);  % word to topic assignment counts accross all documents

theta = zeros(20,20);
word_entropy = zeros(20,20);

% This makes a number of Gibbs sampling sweeps through all docs and words
for iter = 1:20     % This can take a couple of minutes to run
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
  
  theta(:,iter)= (skd(:,2) + alpha) / (sum(skd(:,2) + alpha));
  
  beta = zeros(W,K);
  for o = 1:K
      beta(:,o) = (swk(:,o) + gamma) / (sum(swk(:,o) + gamma));
  end
  
  for i = 1:K
      entropy = 0;
      for j = 1:size(beta,1)
          entropy = entropy + (-beta(j,i)*log(beta(j,i)));
      end
      word_entropy(i,iter) = entropy;
  end
  
end

plot(theta', 'Linewidth', 1);
set(gca,'fontsize',13);
grid on;

title('Topic Posteriors vs Gibbs Iteration', 'FontSize', 20, 'FontWeight', 'bold')
xlabel('Gibbs Iteration', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Topic Posteriors', 'FontSize', 15, 'FontWeight', 'bold');
xlim([1,20])
xticks(1:20)

plot(word_entropy', 'Linewidth', 1);
set(gca,'fontsize',13);
grid on;

title('Word Entropy vs Gibbs Iteration', 'FontSize', 20, 'FontWeight', 'bold')
xlabel('Gibbs Iteration', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Word Entropy', 'FontSize', 15, 'FontWeight', 'bold');
xlim([1,20])
xticks(1:20)