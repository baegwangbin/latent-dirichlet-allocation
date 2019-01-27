% Answer to Question C
% Log probability and per-word perplexity calculated for every document in dataset B

clear all
clc
load kos_doc_data.mat

M = max([A(:,2)]);  % number of unique words
D = max(A(:,1));    % number of documents in A
N = sum(A(:,3));    % number of unique words in A

% calculate beta & log_beta
beta = zeros(1,M);
log_beta = zeros(1,M);
alpha = 0.1;

for m = 1:M
     count_m = sum(A(A(:,2)==m,3));
     beta(m) = (alpha + count_m)/((M*alpha)+N);
     log_beta(m) = log(beta(m));
end

% calculate log_prob and perplexity for every document in B
doc_B = unique(B(:,1));

% first column: document index
% second column: log_prob
% third column: perplexity
results = zeros(size(doc_B,1),4);

for i = 1:size(doc_B,1)
    
    index_i = doc_B(i);
    results(i,1) = index_i;
    
    doc_i = B(find(B(:,1)==index_i),:,:);
    
    log_prob = 0;
    total_count = 0;

    for m = 1:M
        count_m = sum(doc_i(doc_i(:,2)==m,3));
        total_count = total_count + count_m;
        log_prob = log_prob + count_m * log_beta(m);
    end
    
    results(i,2) = log_prob;
    
    perplexity = exp(-log_prob/total_count);
    results(i,3) = perplexity;
    results(i,4) = total_count;

end

plot(results(:,1),results(:,2), 'Linewidth', 1); 
hold on; plot(results(:,1),results(:,3), 'Linewidth', 1);
legend({'Log Probability' 'Perplexity'})

grid on;
set(gca,'fontsize',13);
title('Log Probability and Perplexity', 'FontSize', 20, 'FontWeight', 'bold');
xlabel('Document ID', 'FontSize', 15, 'FontWeight', 'bold');
%ylabel('Words', 'FontSize', 15, 'FontWeight', 'bold');

xlim([2001 3430]);





