% Answer to Question C
% Calculate per-word perplexity of document 2001

clear all
clc
load kos_doc_data.mat

M = max([A(:,2)]);  % number of unique words
D = max(A(:,1));    % number of documents in A

index_2001 = find(B(:,1)==2001);
uniq_2001 = unique(B(index_2001,2));
doc_2001 = B(index_2001,:,:);

uniq_A = unique(A(:,2));
uniq_B = unique(B(:,2));

%setdiff(uniq_2001, uniq_A);
%setdiff(uniq_B, uniq_A);


N = sum(A(:,3));
beta = zeros(1,M);
log_beta = zeros(1,M);
alpha = 0.1;

for m = 1:M
     count_m = sum(A(A(:,2)==m,3));
     beta(m) = (alpha + count_m)/((M*alpha)+N);
     log_beta(m) = log(beta(m));
end


log_prob = 0;
total_count = 0;

for m = 1:M
    count_m = sum(doc_2001(doc_2001(:,2)==m,3));
    total_count = total_count + count_m;
    log_prob = log_prob + count_m * log_beta(m);
end

log_prob
total_count
perplexity = exp(-log_prob/total_count)


